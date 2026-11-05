# Native binary handoff

PR: https://github.com/MissyLabs/missy/pull/171

This build is a native Linux x86_64 GNU ELF `missy-rs`, not a Python launcher.
It does not replace the installed Python `missy` command or start any service.
Use separate config/state, explicit provider/policy grants and dedicated test
credentials. Start with `missy-rs --help`, `init` and `config-check`; do not reuse
a live Python database or bot token. Read `README-RUST.md` and the parity ledger
before enabling optional capabilities or promoting it.

The locally accepted executable has SHA-256:

`91f7d332d3502696de7ad006cdb6da9865475a3ffa11f7697bbaa0f930bcb5a1`

This is not a static binary. On compatible Debian/Ubuntu Linux it requires
`libopus0`, glibc, libm, libgcc_s and the system ELF loader. Build requires a
stable Rust toolchain and native Opus prerequisites (`libopus-dev pkg-config`).
Local production-conversion acceptance used Linux Mint/Ubuntu glibc and Rust 1.99.1, not a claim of
compatibility with every distribution or musl. SQLite and Rustls are compiled in.

Discord DAVE voice additionally requires an explicitly built and operator-pinned
official library, source/version/hash and approved endpoints. No library is
silently downloaded or loaded by name. See `crates/missy-dave/README.md` and
`crates/missy-discord-voice/README.md`; the dedicated CI artifact retains native
provenance and licenses. Speech executables, FFmpeg, camera/desktop endpoints,
local embeddings and infrastructure services are separately configured, not
bundled or automatically enabled.

Cargo metadata declares MIT for the rewrite, matching legacy project metadata.
Native dependencies retain their licenses: Opus BSD-style notice is included in
the local binary bundle; official DAVE/MLS/native crypto license and provenance
details are in its separately built artifact. Do not remove third-party notices
when redistributing. System libraries remain distribution-maintained dependencies.

Final local evidence: 740 ordinary passing entries, all 3 separately enabled
native cases passed (743 distinct total), strict format/lint/dependency audit,
release build and 15 Chromium assertions. Test totals include helper entrypoints,
not independent end-to-end scenarios. No live Discord/language-provider canary,
hardware capture/playback, GPU job, deployment or production migration occurred.

GitHub Actions `Rust rewrite` publishes the executable and native DAVE artifacts.
Select a successful run for the exact PR head; older green checks do not certify
new changes. The SHA above identifies this local build, not bit-identical builds
on other compilers/runners. `production-validation.md` records the final integrated
conversion evidence; `production-replacement-matrix.md` records exact behavior
dispositions and intentional API/security differences. Earlier `validation.md`
checkpoints describe older commits.
