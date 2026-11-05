#!/usr/bin/env bash
# Explicit operator build. Never invoked by Cargo or the application.
set -euo pipefail
umask 022
root="${1:?usage: bash build-native.sh /absolute/isolated/build-cache}"
[[ "$root" = /* ]] || { echo 'Absolute isolated build-cache path required' >&2; exit 2; }
for tool in git cmake ninja c++ pkg-config openssl; do command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool (Debian: git cmake ninja-build build-essential pkg-config libssl-dev)" >&2; exit 2; }; done
pkg-config --atleast-version=3.0 openssl || { echo 'Distribution-maintained OpenSSL 3 development libraries required' >&2; exit 2; }
mkdir -p "$root"
cd "$root"
# mlspp forcibly enables CMake's user package registry internally. Isolate HOME
# as well as the install prefix, so even that registry stays inside this cache.
export HOME="$root/build-home"
mkdir -p "$HOME"
clone_pin() {
    local repo="$1" dir="$2" commit="$3"
    if [[ ! -d "$dir/.git" ]]; then git clone "$repo" "$dir"; fi
    [[ -z "$(git -C "$dir" status --porcelain)" ]] || { echo "Dirty source refused: $dir" >&2; exit 2; }
    git -C "$dir" checkout --detach "$commit"
    [[ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ]]
}
clone_pin https://github.com/discord/libdave.git libdave-1.2.2 6396c90aedbb9a6ee117ea9ce1ce9832af6a3375
clone_pin https://github.com/cisco/mlspp.git mlspp 1cc50a124a3bc4e143a787ec934280dc70c1034d
clone_pin https://github.com/nlohmann/json.git json 55f93686c01528224f448c19128836e7df245f72
clone_pin https://github.com/google/googletest.git googletest 52eb8108c5bdec04579160ae17225d66034bd723
# vcpkg gitlinks are pinned upstream but unused: system OpenSSL + local prefix,
# no bootstrap script, no system install, no CMake user-package registration.
[[ "$(git -C libdave-1.2.2 rev-parse HEAD:cpp/vcpkg)" = 16c71a39e5a0fc0bdb3fad03beef8f38ee00ee3b ]]
[[ "$(git -C mlspp rev-parse HEAD:vcpkg)" = acd65983f2e668c0bc44a117575e6073681fa8e3 ]]
cmake -S json -B json-build -G Ninja -DJSON_BuildTests=OFF -DCMAKE_INSTALL_PREFIX="$root/prefix" -DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF
cmake --install json-build
cmake -S googletest -B googletest-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="$root/prefix"
cmake --build googletest-build -j 4
cmake --install googletest-build
cmake -S mlspp -B mlspp-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_PREFIX_PATH="$root/prefix" -DCMAKE_INSTALL_PREFIX="$root/prefix" -DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF -DMLS_CXX_NAMESPACE=mlspp -DDISABLE_GREASE=ON
cmake --build mlspp-build -j 4
cmake --install mlspp-build
cmake -S libdave-1.2.2/cpp -B libdave-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$root/prefix" -DCMAKE_INSTALL_PREFIX="$root/prefix" -DBUILD_SHARED_LIBS=ON -DTESTING=ON
cmake --build libdave-build -j 4
# Root-level upstream ctest finds zero tests: execute actual suites explicitly.
"$root/libdave-build/test/libdave_test"
"$root/libdave-build/test/capi/capi_test"
sha256sum "$root/libdave-build/libdave.so"
openssl version -a
ldd "$root/libdave-build/libdave.so"
mkdir -p "$root/artifact/licenses"
cp "$root/libdave-build/libdave.so" "$root/artifact/libdave.so"
cp "$root/libdave-1.2.2/LICENSE" "$root/artifact/licenses/libdave-MIT.txt"
cp "$root/mlspp/LICENSE" "$root/artifact/licenses/mlspp-BSD-2-Clause.txt"
cp "$root/json/LICENSE.MIT" "$root/artifact/licenses/nlohmann-json-MIT.txt"
# OpenSSL remains a distribution dependency, not an untracked vendored binary.
if [[ -f /usr/share/doc/libssl3t64/copyright ]]; then
    cp /usr/share/doc/libssl3t64/copyright "$root/artifact/licenses/openssl-distribution-copyright.txt"
elif [[ -f /usr/share/doc/libssl3/copyright ]]; then
    cp /usr/share/doc/libssl3/copyright "$root/artifact/licenses/openssl-distribution-copyright.txt"
fi
(cd "$root/artifact" && sha256sum libdave.so > SHA256SUMS)
{
    printf 'libdave=6396c90aedbb9a6ee117ea9ce1ce9832af6a3375\nmlspp=1cc50a124a3bc4e143a787ec934280dc70c1034d\n'
    openssl version -a
    pkg-config --modversion openssl
    c++ --version
    ldd "$root/artifact/libdave.so"
} > "$root/artifact/BUILD-PROVENANCE.txt"
