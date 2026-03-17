#!/usr/bin/env bash
# install.sh — One-line installer for Missy
# Usage: curl -fsSL https://raw.githubusercontent.com/MissyLabs/missy/master/install.sh | bash
set -euo pipefail

REPO="https://github.com/MissyLabs/missy.git"
INSTALL_DIR="$HOME/.local/share/missy"
VENV_DIR="$INSTALL_DIR/.venv"
BIN_DIR="$HOME/.local/bin"

echo "=== Installing Missy ==="
echo

# ── Check Python ──
PYTHON=""
for py in python3.12 python3.11 python3; do
    if command -v "$py" &>/dev/null; then
        ver="$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
        major="${ver%%.*}"
        minor="${ver##*.}"
        if (( major >= 3 && minor >= 11 )); then
            PYTHON="$py"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "Error: Python 3.11+ is required." >&2
    echo "Install it with: sudo apt install python3.11  (or your package manager)" >&2
    exit 1
fi
echo "Using $PYTHON ($($PYTHON --version))"

# ── Check git ──
if ! command -v git &>/dev/null; then
    echo "Error: git is required." >&2
    echo "Install it with: sudo apt install git" >&2
    exit 1
fi

# ── Clone or update ──
if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "Updating existing install at $INSTALL_DIR..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "Cloning missy to $INSTALL_DIR..."
    git clone "$REPO" "$INSTALL_DIR"
fi

# ── Create venv and install ──
echo "Setting up virtual environment..."
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$INSTALL_DIR" -q

# ── Symlink binary ──
mkdir -p "$BIN_DIR"
ln -sf "$VENV_DIR/bin/missy" "$BIN_DIR/missy"

# ── Check PATH ──
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo
    echo "Add this to your shell profile (~/.bashrc or ~/.zshrc):"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    echo
fi

# ── Run setup ──
echo
echo "=== Missy installed ==="
echo
echo "Run the setup wizard:"
echo "  missy setup"
echo
echo "Or jump straight in:"
echo "  export ANTHROPIC_API_KEY=\"sk-ant-...\""
echo "  missy ask \"Hello\""
echo
echo "Optional extras:"
echo "  $VENV_DIR/bin/pip install -e \"$INSTALL_DIR[voice]\"   # voice channel"
echo "  $VENV_DIR/bin/pip install -e \"$INSTALL_DIR[otel]\"    # OpenTelemetry"
