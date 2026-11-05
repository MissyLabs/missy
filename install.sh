#!/usr/bin/env bash
# Native Rust installation. Never reads legacy state or starts a service.
set -euo pipefail
umask 077
prefix="${HOME:?}/.local/share/missy-native"
bin_dir="$HOME/.local/bin"
ref=rewrite/rust-runtime
jobs=2
source_dir=
upgrade=0
usage() { printf '%s\n' 'install.sh [--ref BRANCH_OR_COMMIT] [--source CHECKOUT] [--prefix ABSOLUTE_PATH] [--bin-dir ABSOLUTE_PATH] [--jobs N] [--upgrade]' >&2; exit 2; }
while (($#)); do
  case "$1" in
    --ref) (($#>=2)) || usage; ref=$2; shift 2;;
    --source) (($#>=2)) || usage; source_dir=$2; shift 2;;
    --prefix) (($#>=2)) || usage; prefix=$2; shift 2;;
    --bin-dir) (($#>=2)) || usage; bin_dir=$2; shift 2;;
    --jobs) (($#>=2)) || usage; jobs=$2; shift 2;;
    --upgrade) upgrade=1; shift;;
    --help|-h) usage;;
    *) usage;;
  esac
done
[[ "$prefix" = /* && "$bin_dir" = /* && "$prefix" != / && "$bin_dir" != / && "$jobs" =~ ^[1-9][0-9]?$ ]] || usage
[[ "$prefix" != *'/../'* && "$prefix" != */.. && "$bin_dir" != *'/../'* && "$bin_dir" != */.. ]] || usage
[[ ! -L "$prefix" && ! -L "$bin_dir" ]] || { echo 'Refusing symlink installation directories.' >&2; exit 1; }
[[ -n "$ref" && "$ref" != -* && "$ref" != *$'\n'* ]] || usage
for command in cargo rustc git sha256sum flock install; do
  command -v "$command" >/dev/null || { printf 'Missing dependency: %s\n' "$command" >&2; exit 1; }
done
if ! pkg-config --exists opus 2>/dev/null; then
  echo 'Build requires pkg-config and libopus development files (Debian/Ubuntu: libopus-dev pkg-config).' >&2; exit 1
fi
mkdir -p "$prefix" "$bin_dir"
prefix=$(realpath -e "$prefix")
bin_dir=$(realpath -e "$bin_dir")
destination="$bin_dir/missy-rs"
if [[ -e "$destination" || -L "$destination" ]]; then
  [[ "$upgrade" = 1 && -L "$destination" && $(readlink "$destination") = "$prefix/current/missy-rs" ]] || {
    echo 'Existing command is not replaced. --upgrade only accepts this native installation.' >&2; exit 1;
  }
fi
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
if [[ -n "$source_dir" ]]; then
  source_dir=$(realpath -e "$source_dir")
  [[ -z $(git -C "$source_dir" status --porcelain) ]] || { echo 'Local source must be a clean committed checkout.' >&2; exit 1; }
  git clone --no-hardlinks --no-checkout "$source_dir" "$work/source"
  revision=$(git -C "$source_dir" rev-parse HEAD)
else
  git clone --filter=blob:none --no-checkout https://github.com/MissyLabs/missy.git "$work/source"
  # A no-checkout clone creates only the default local branch. Resolve selected
  # branches through origin, then tags/commits. Never silently build master.
  if git -C "$work/source" show-ref --verify --quiet "refs/remotes/origin/$ref"; then
    revision=$(git -C "$work/source" rev-parse --verify "refs/remotes/origin/$ref^{commit}")
  else
    revision=$(git -C "$work/source" rev-parse --verify "$ref^{commit}")
  fi
fi
git -C "$work/source" checkout --detach "$revision"
[[ -f "$work/source/Cargo.lock" && -f "$work/source/packaging/rust/release.sh" ]] || { echo 'Selected revision is not a native release.' >&2; exit 1; }
(cd "$work/source" && CARGO_BUILD_JOBS="$jobs" cargo build --release --locked -p missy-cli)
binary="$work/source/target/release/missy-rs"
digest=$(sha256sum "$binary" | cut -d' ' -f1)
version="$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:12}"
bash "$work/source/packaging/rust/release.sh" stage "$prefix" "$version" "$binary" "$digest"
bash "$work/source/packaging/rust/release.sh" activate "$prefix" "$version"
if [[ ! -L "$destination" ]]; then ln -s "$prefix/current/missy-rs" "$destination"; fi
printf '\nNative executable: %s\nSource revision: %s\nSHA-256: %s\n' "$destination" "$revision" "$digest"
printf '%s\n' 'No account, config, legacy state or running process was changed.' 'Next: missy-rs --config /your/private/config.yaml init' 'Read README-RUST.md before enabling services.'
