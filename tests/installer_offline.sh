#!/usr/bin/env bash
# Disposable end-to-end test: bash tests/installer_offline.sh
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
test_root=$(mktemp -d)
trap 'chmod -R u+w "$test_root" 2>/dev/null || :; rm -rf -- "$test_root"' EXIT
export HOME="$test_root/home" TMPDIR="$test_root/tmp" GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2
mkdir -p "$HOME" "$TMPDIR" "$test_root/wrappers" "$test_root/fixture"
real_git=$(command -v git)
export REAL_GIT="$real_git" FIXTURE_REPO="$test_root/fixture"
cat > "$test_root/wrappers/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1-} == clone ]]; then
  args=()
  for arg in "$@"; do
    if [[ "$arg" == https://github.com/MissyLabs/missy.git ]]; then
      args+=("$FIXTURE_REPO")
    elif [[ "$arg" == *://* || "$arg" == git@* ]]; then
      echo 'Unexpected network clone' >&2; exit 1
    else args+=("$arg"); fi
  done
  exec "$REAL_GIT" "${args[@]}"
fi
case " $* " in *' fetch '*|*' pull '*|*' push '*) echo 'Network git operation forbidden' >&2; exit 1;; esac
exec "$REAL_GIT" "$@"
EOF
cat > "$test_root/wrappers/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${FAIL_BUILD-0} != 1 ]] || { echo 'Injected build failure' >&2; exit 1; }
[[ "$*" == 'build --release --locked -p missy-cli' && ${CARGO_BUILD_JOBS-} == 2 ]]
mkdir -p target/release
cp fixture-binary target/release/missy-rs
chmod +x target/release/missy-rs
EOF
printf '#!/bin/sh\nexit 0\n' > "$test_root/wrappers/pkg-config"
chmod +x "$test_root/wrappers/"*
export PATH="$test_root/wrappers:$PATH"

git -C "$FIXTURE_REPO" init -q -b master
git -C "$FIXTURE_REPO" config user.name 'Offline fixture'
git -C "$FIXTURE_REPO" config user.email offline@example.invalid
mkdir -p "$FIXTURE_REPO/packaging/rust"
cp "$repo/packaging/rust/release.sh" "$FIXTURE_REPO/packaging/rust/release.sh"
touch "$FIXTURE_REPO/Cargo.lock"
make_binary() {
  printf '#!/bin/sh\nprintf "%%s\\n" "%s"\n' "$1" > "$FIXTURE_REPO/fixture-binary"
}
make_binary master
git -C "$FIXTURE_REPO" add .
git -C "$FIXTURE_REPO" commit -qm master
git -C "$FIXTURE_REPO" checkout -qb rewrite/rust-runtime
make_binary rust-default
git -C "$FIXTURE_REPO" commit -qam rust-default
default_revision=$(git -C "$FIXTURE_REPO" rev-parse HEAD)
git -C "$FIXTURE_REPO" checkout -qb feature/alternate
make_binary rust-alternate
git -C "$FIXTURE_REPO" commit -qam rust-alternate
alternate_revision=$(git -C "$FIXTURE_REPO" rev-parse HEAD)
git -C "$FIXTURE_REPO" checkout -q master

prefix="$test_root/native" bin="$test_root/bin"
mkdir -p "$HOME/.config/missy" "$HOME/.local/share/missy"
printf 'legacy config\n' > "$HOME/.config/missy/config.yaml"
printf 'legacy state\n' > "$HOME/.local/share/missy/state"
legacy_before=$(tar -C "$HOME" -cf - .config/missy .local/share/missy | sha256sum)
install_run() { "$repo/install.sh" --prefix "$prefix" --bin-dir "$bin" "$@"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
expect_failure() { if "$@" >"$test_root/failure.log" 2>&1; then fail "unexpected success: $*"; fi; }

[[ -x "$repo/install.sh" ]] || fail 'root installer is not executable'
install_run > "$test_root/default.log"
grep -Fq "Source revision: $default_revision" "$test_root/default.log" || fail 'wrong default revision'
[[ $("$bin/missy-rs" --version) == rust-default ]] || fail 'wrong default executable'
[[ -L "$bin/missy-rs" && $(readlink "$bin/missy-rs") == "$prefix/current/missy-rs" ]] || fail 'bad command symlink'
first=$(readlink "$prefix/current")
[[ "$first" == releases/* && -d "$prefix/$first" ]] || fail 'invalid release path'
# Check mode bits rather than -w, which is misleading when run as root.
[[ $(stat -c %a "$prefix/$first/missy-rs") == 555 && $(stat -c %a "$prefix/$first") == 555 ]] || fail 'release not immutable'
first_digest=$(sha256sum "$prefix/$first/missy-rs")
expect_failure install_run
[[ $(readlink "$prefix/current") == "$first" ]] || fail 'non-upgrade changed current'
install_run --upgrade --ref feature/alternate > "$test_root/upgrade.log"
grep -Fq "Source revision: $alternate_revision" "$test_root/upgrade.log" || fail 'wrong explicit branch revision'
[[ $("$bin/missy-rs" --version) == rust-alternate ]] || fail 'wrong upgrade executable'
second=$(readlink "$prefix/current")
[[ "$second" != "$first" && $(readlink "$prefix/previous") == "$first" ]] || fail 'upgrade links incorrect'
[[ $(sha256sum "$prefix/$first/missy-rs") == "$first_digest" ]] || fail 'old release modified'
expect_failure env FAIL_BUILD=1 "$repo/install.sh" --prefix "$prefix" --bin-dir "$bin" --upgrade
grep -q 'Injected build failure' "$test_root/failure.log" || fail 'failure did not reach build'
[[ $(readlink "$prefix/current") == "$second" && $("$bin/missy-rs" --version) == rust-alternate ]] || fail 'failed build changed active release'

rm "$bin/missy-rs"
printf 'foreign command\n' > "$bin/missy-rs"
expect_failure install_run --upgrade
[[ $(cat "$bin/missy-rs") == 'foreign command' && $(readlink "$prefix/current") == "$second" ]] || fail 'foreign file overwritten'
rm "$bin/missy-rs"
ln -s "$test_root/foreign" "$bin/missy-rs"
expect_failure install_run --upgrade
[[ $(readlink "$bin/missy-rs") == "$test_root/foreign" ]] || fail 'foreign symlink overwritten'

release="$repo/packaging/rust/release.sh"
digest=$(sha256sum "$prefix/$first/missy-rs" | cut -d' ' -f1)
expect_failure bash "$release" stage "$prefix" "${first#releases/}" "$prefix/$first/missy-rs" "$digest"
[[ $(sha256sum "$prefix/$first/missy-rs") == "$first_digest" ]] || fail 'duplicate stage changed release'
expect_failure bash "$release" stage "$prefix" invalid-checksum "$prefix/$first/missy-rs" "$(printf '0%.0s' {1..64})"
[[ ! -e "$prefix/releases/invalid-checksum" && $(readlink "$prefix/current") == "$second" ]] || fail 'bad stage published'
bash "$release" rollback "$prefix" > /dev/null
[[ $(readlink "$prefix/current") == "$first" && $(readlink "$prefix/previous") == "$second" ]] || fail 'rollback links incorrect'
legacy_after=$(tar -C "$HOME" -cf - .config/missy .local/share/missy | sha256sum)
[[ "$legacy_before" == "$legacy_after" ]] || fail 'legacy data changed'
echo 'PASS: offline installer, releases, upgrade, refusal, failure and rollback'
