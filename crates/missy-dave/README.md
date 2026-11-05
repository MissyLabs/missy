# Official native DAVE backend

This MIT Rust crate wraps the **official Discord libdave C API**, rather than
implementing cryptography or using davey/OpenMLS/RustCrypto. It supports Linux
only for now. No AGPL code, advisory ignore, cargo patch, or license change is
used. All build actions below are explicit operator/developer commands, never
an application startup or Cargo build script.

## Pins and licenses

| Component | Exact revision | License |
|---|---|---|
| Discord libdave `v1.2.2/cpp` | `6396c90aedbb9a6ee117ea9ce1ce9832af6a3375` | MIT |
| Cisco mlspp (official libdave overlay selection) | `1cc50a124a3bc4e143a787ec934280dc70c1034d` | BSD-2-Clause |
| nlohmann/json `v3.12.0` | `55f93686c01528224f448c19128836e7df245f72` | MIT |
| GoogleTest `v1.17.0`, tests only | `52eb8108c5bdec04579160ae17225d66034bd723` | BSD-3-Clause |
| OpenSSL, host maintained | locally tested `3.0.13-0ubuntu3.16` | Apache-2.0 |

libdave's vcpkg gitlink is `16c71a39e5a0fc0bdb3fad03beef8f38ee00ee3b`;
mlspp's is `acd65983f2e668c0bc44a117575e6073681fa8e3`. Neither is needed by
this build; the script verifies both pins and does not bootstrap vcpkg.
No submodule's source is included in the resulting library by this workflow.
Retain upstream `LICENSE` files when redistributing native binaries and include
the license texts of their linked dependencies. This repository does not ship
the native binary or third-party source tree.

## Explicit reproducible-source build

Prerequisites: Git, CMake, Ninja, a C++17 compiler, and distribution-maintained
OpenSSL 3 development libraries. Nothing is installed globally. Do not use an
unpatched upstream OpenSSL tarball simply because its version starts with 3.

```sh
bash crates/missy-dave/build-native.sh /absolute/isolated/build-cache
```

All CMake installs go to that cache's `prefix/`; HOME and CMake user registries
are isolated under its `build-home/`. The distribution-ready library, license
notices, SHA file, compiler/crypto/dependency evidence go to `artifact/`.
Source revisions are pinned;
the final binary SHA depends on compiler, OS, OpenSSL ABI, and build directory,
so this is not a claim of bit-for-bit reproducibility across machines.
The local verified build is in `/home/odin/projects/missy-rust-deps/`.
At verification time `libdave-build/libdave.so` had SHA-256
`1ced0809cbdccb64a5429a8ee28df00e4082b5fc6299262b7bb64b085bdee289`.

## Runtime configuration and trust boundary

`NativeDaveConfig` requires all four fields with no defaults:

```json
{
  "library_path": "/absolute/operator-approved/libdave.so",
  "sha256": "64-hex-digit-operator-approved-build-digest",
  "version": "1.2.2/cpp",
  "source_commit": "6396c90aedbb9a6ee117ea9ce1ce9832af6a3375"
}
```

Wrong source/version, relative path, symlink, missing library, wrong hash,
missing C API symbol, or incompatible maximum protocol version fails closed.
The loader snapshots at most 64 MiB and hashes the exact bytes subsequently
loaded from a sealed Linux memfd, preventing path-swap/in-place-write races.
It never searches system library names. The operator must independently obtain
the expected SHA from their trusted build; putting any file's hash in config
does not authenticate its origin. The official C API has no release-version
query, so source version provenance comes from the trusted build, not the
maximum DAVE protocol query. Native code is executable trusted code.

The library is dynamically linked to host `libcrypto.so.3`, libc, libm, libgcc,
and C++ runtime. **The top-level SHA does not attest system dependencies.**
Distribution security updates and package provenance remain the operator's
responsibility. Cargo audit checks Rust dependencies, not native
OpenSSL/mlspp/libdave vulnerabilities. No blanket native security assertion is
made. The local OpenSSL package was built 2026-09-16; its upstream display
version alone does not describe downstream security backports.

## Tests and actual evidence

Normal `cargo test -p missy-dave` tests missing/unpinned library rejection and
hash verification before executing code. One native integration test is
explicitly ignored by default because ordinary Cargo tests must not download,
compile, or execute externally supplied libraries automatically.

Run it against an explicitly trusted native build:

```sh
MISSY_TEST_LIBDAVE_PATH=/absolute/cache/libdave-build/libdave.so \
MISSY_TEST_LIBDAVE_SHA256=<trusted-sha256> \
MISSY_TEST_LIBDAVE_HELPER=/absolute/cache/libdave-build/test/capi/libexternal_sender.so \
cargo test -p missy-dave -- --include-ignored
```

The helper is the **upstream external sender**, not invented/mock cryptography.
The helper path is explicit and test-only; unlike the production loader this
developer test does not hash the helper before loading it.

Verified 2026-11-05 locally:

* Official C++ suite: **62 tests passed**.
* Official C API suite: **9 tests passed**, including two real MLS sessions,
  proposal/commit/welcome, matching epoch authenticators and media roundtrip.
* Rust wrapper: **3 tests passed including the explicitly enabled native test**.
  The wrapper test establishes two independent MLS sessions through that
  upstream external sender, installs transitions, encrypts/decrypts both ways,
  rejects replay, tampering, plaintext downgrade, unknown sender, protocol
  downgrade, malformed external sender, and encryption before transition.

No Discord login, network voice session, microphone, speaker, recording, or
live hardware was used. Test media are synthetic bytes, not an Opus codec or
audio-quality test. The official DAVE encryptor treats Opus frames as opaque.

## Implementation details

The production wrapper calls the C API directly, with RAII handles and native
output-buffer release via `daveFree`. Ratchets are copied by libdave when
installed, then the caller's handles are destroyed. Pending transitions retain
their ratchets separately from the live session, capped at 16. Decryptors
retain upstream replay state across epoch transitions. Passthrough is never
enabled; native MLS errors are converted to generic errors without logging
key material. There is no custom crypto fixture, FFI process proxy, default
native-library path, native download, subprocess or Python runtime.
