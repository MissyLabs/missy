//! Deliberately narrow unified diff grammar: tracked regular text modifications.
//! No fuzzy matching, binary data, modes, renames, additions, deletions or paths
//! requiring Git quoting. Unsupported input fails before any mutation.
use crate::{
    FileChange,
    repository::{EvolutionPolicy, Repository},
};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

pub(crate) fn path_valid(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 1024
        && path.split('/').all(|c| {
            !c.is_empty()
                && c != "."
                && c != ".."
                && !c.eq_ignore_ascii_case(".git")
                && c.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
}
pub(crate) fn prepare(
    diff: &str,
    repo: &Repository,
    policy: &EvolutionPolicy,
) -> Result<Vec<FileChange>> {
    ensure!(
        !diff.is_empty()
            && diff.len() <= 1024 * 1024
            && !diff.contains('\0')
            && diff.ends_with('\n'),
        "bounded newline-terminated text diff required"
    );
    let lines: Vec<&str> = diff.split_inclusive('\n').collect();
    let mut i = 0;
    let mut seen = BTreeSet::new();
    let mut out = vec![];
    while i < lines.len() {
        let header = lines[i].trim_end_matches('\n');
        let rest = header
            .strip_prefix("diff --git a/")
            .context("expected diff --git header")?;
        let (a, b) = rest.split_once(" b/").context("invalid diff paths")?;
        ensure!(
            a == b && path_valid(a) && seen.insert(a.to_owned()),
            "unsafe/duplicate/mismatched patch path"
        );
        i += 1;
        if i < lines.len() && lines[i].starts_with("index ") {
            let parts: Vec<_> = lines[i].trim_end().split(' ').collect();
            ensure!(
                parts.len() == 2 || (parts.len() == 3 && ["100644", "100755"].contains(&parts[2])),
                "unsupported index/mode"
            );
            i += 1;
        }
        ensure!(
            lines.get(i) == Some(&format!("--- a/{a}\n").as_str())
                && lines.get(i + 1) == Some(&format!("+++ b/{a}\n").as_str()),
            "only identical existing-file headers supported"
        );
        i += 2;
        policy.check_file(&repo.path, a)?;
        let (before, mode) = repo.read_file(a)?;
        ensure!(
            before.len() <= 4 * 1024 * 1024 && !before.contains('\0') && before.ends_with('\n'),
            "regular newline-terminated bounded UTF-8 source required"
        );
        let original: Vec<&str> = before.split_inclusive('\n').collect();
        let mut after = String::new();
        let mut cursor = 0usize;
        let mut hunks = 0;
        while i < lines.len() && lines[i].starts_with("@@ ") {
            let header = lines[i].trim_end_matches('\n');
            let inner = header
                .strip_prefix("@@ -")
                .and_then(|s| s.split_once(" @@"))
                .context("invalid hunk header")?
                .0;
            let (old, new) = inner.split_once(" +").context("invalid hunk ranges")?;
            fn range(s: &str) -> Result<(usize, usize)> {
                let (start, count) = s.split_once(',').unwrap_or((s, "1"));
                Ok((start.parse()?, count.parse()?))
            }
            let (old_start, old_count) = range(old)?;
            let (new_start, new_count) = range(new)?;
            ensure!(
                old_start > 0 && new_start > 0,
                "zero-start hunks unsupported"
            );
            let start = old_start - 1;
            ensure!(
                start >= cursor && start <= original.len(),
                "overlapping/out-of-range hunk"
            );
            for l in &original[cursor..start] {
                after.push_str(l);
            }
            ensure!(
                after.bytes().filter(|b| *b == b'\n').count() == new_start - 1,
                "new hunk position mismatch"
            );
            cursor = start;
            i += 1;
            let mut removed = 0;
            let mut added = 0;
            while i < lines.len()
                && !lines[i].starts_with("@@ ")
                && !lines[i].starts_with("diff --git ")
            {
                let line = lines[i];
                ensure!(
                    matches!(line.as_bytes().first(), Some(b' ' | b'-' | b'+')),
                    "unsupported patch line"
                );
                let payload = &line[1..];
                match line.as_bytes()[0] {
                    b' ' => {
                        ensure!(original.get(cursor) == Some(&payload), "context mismatch");
                        after.push_str(payload);
                        cursor += 1;
                        removed += 1;
                        added += 1;
                    }
                    b'-' => {
                        ensure!(
                            original.get(cursor) == Some(&payload),
                            "removed content mismatch"
                        );
                        cursor += 1;
                        removed += 1;
                    }
                    b'+' => {
                        after.push_str(payload);
                        added += 1;
                    }
                    _ => anyhow::bail!("unsupported patch line"),
                }
                i += 1;
            }
            ensure!(
                removed == old_count && added == new_count,
                "hunk count mismatch"
            );
            hunks += 1;
        }
        ensure!(hunks > 0, "missing hunks");
        for l in &original[cursor..] {
            after.push_str(l);
        }
        ensure!(
            before != after && after.len() <= 4 * 1024 * 1024,
            "empty/oversized change"
        );
        out.push(FileChange {
            path: a.to_owned(),
            before,
            after,
            mode,
        });
        ensure!(out.len() <= 64, "at most 64 files per transaction");
    }
    ensure!(!out.is_empty(), "no files");
    Ok(out)
}
