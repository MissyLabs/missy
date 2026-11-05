#!/bin/sh
# Never invokes a service manager, edits configuration, or touches runtime state.
set -eu
umask 077
usage() { echo 'release.sh stage PREFIX VERSION BINARY SHA256 | activate PREFIX VERSION | rollback PREFIX | inspect PREFIX' >&2; exit 2; }
[ "$#" -ge 2 ] || usage
action=$1 prefix=$2
case "$prefix" in /*) ;; *) exit 2;; esac
case "$prefix" in /|*/../*|*/..|*/./*|*/.) exit 2;; esac
[ ! -L "$prefix" ] || exit 1
mkdir -p "$prefix/releases"
[ ! -L "$prefix/releases" ] || exit 1
exec 9>"$prefix/.release.lock"
flock -x 9
version_ok() { case "$1" in ''|*[!A-Za-z0-9._-]*|.|..) exit 2;; esac; }
target_ok() {
    case "$1" in releases/*) version_ok "${1#releases/}";; *) exit 1;; esac
    [ -f "$prefix/$1/missy-rs" ] && [ ! -L "$prefix/$1" ] || exit 1
    (cd "$prefix/$1" && sha256sum -c SHA256SUMS >/dev/null)
}
link_current() {
    target_ok "$1"
    tmp="$prefix/.current.$$"
    [ ! -e "$tmp" ] && [ ! -L "$tmp" ] || exit 1
    ln -s "$1" "$tmp"
    mv -Tf "$tmp" "$prefix/current"
}
case "$action" in
stage)
    [ "$#" -eq 5 ] || usage
    version=$3 binary=$4 expected=$5
    version_ok "$version"
    [ -f "$binary" ] && [ ! -L "$binary" ] || exit 1
    case "$expected" in *[!a-fA-F0-9]*|'') exit 2;; esac
    [ "${#expected}" -eq 64 ] || exit 2
    dest="$prefix/releases/$version"
    [ ! -e "$dest" ] && [ ! -L "$dest" ] || exit 1
    tmp=$(mktemp -d "$prefix/releases/.staging.XXXXXXXX")
    trap 'chmod u+w "$tmp" 2>/dev/null || :; rm -rf -- "$tmp"' EXIT HUP INT TERM
    install -m 0755 "$binary" "$tmp/missy-rs"
    printf '%s  missy-rs\n' "$expected" > "$tmp/SHA256SUMS"
    (cd "$tmp" && sha256sum -c SHA256SUMS >/dev/null && ./missy-rs --version)
    chmod 0555 "$tmp/missy-rs" "$tmp"
    mv -T "$tmp" "$dest"
    trap - EXIT HUP INT TERM
    echo "Staged $version. No config, state, process or service changed."
    ;;
activate)
    [ "$#" -eq 3 ] || usage
    version_ok "$3"
    target="releases/$3"
    target_ok "$target"
    if [ -L "$prefix/current" ]; then
        old=$(readlink "$prefix/current"); target_ok "$old"
        if [ "$old" != "$target" ]; then ln -s "$old" "$prefix/.previous.$$"; mv -Tf "$prefix/.previous.$$" "$prefix/previous"; fi
    elif [ -e "$prefix/current" ]; then exit 1; fi
    link_current "$target"
    echo "Selected $3. Running processes unchanged. Validate config and authorize restart separately."
    ;;
rollback)
    [ "$#" -eq 2 ] || usage
    [ -L "$prefix/previous" ] && [ -L "$prefix/current" ] || exit 1
    previous=$(readlink "$prefix/previous") current=$(readlink "$prefix/current")
    target_ok "$previous"; target_ok "$current"
    link_current "$previous"
    ln -s "$current" "$prefix/.previous.$$"
    mv -Tf "$prefix/.previous.$$" "$prefix/previous"
    echo 'Selected previous binary only. No database/config rollback; running processes unchanged.'
    ;;
inspect)
    [ "$#" -eq 2 ] || usage
    for name in current previous; do
        if [ -L "$prefix/$name" ]; then target=$(readlink "$prefix/$name"); target_ok "$target"; printf '%s %s\n' "$name" "$target"; fi
    done
    ;;
*) usage;;
esac
