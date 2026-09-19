#!/usr/bin/env bash
#
# Runs on the HOST (devcontainer.json -> initializeCommand), before the
# container is created or started.
#
# Makes a private COPY of the host's shell history and SSH files in a staging
# directory, which devcontainer.json then bind-mounts READ-ONLY into the
# container. The originals are only ever read:
#
#   ~/.zsh_history   --cp-->  <stage>/zsh_history
#   ~/.ssh/*         --cp-->  <stage>/ssh/*
#
# The container never sees ~/.ssh or ~/.zsh_history themselves, so nothing it
# does can modify them.
#
# The stage lives in ~/.cache, outside the repository, so keys can never be
# committed by accident. Written for bash 3.2 so it runs on stock macOS.

set -u
umask 077

: "${HOME:?HOME is not set}"
STAGE="${HOME}/.cache/ai4se-devcontainer/host-seed"

mkdir -p "${STAGE}"
chmod 700 "${STAGE}"

# Empty the stage but keep the directory itself: if a container is already
# running, its bind mount points at this directory and must stay valid.
find "${STAGE}" -mindepth 1 -delete 2>/dev/null || true

# Lets the container rewrite absolute host paths in ~/.ssh/config
# (e.g. IdentityFile /Users/alice/.ssh/id_ed25519).
printf '%s\n' "${HOME}" > "${STAGE}/host_home"

# ------------------------------------------------------------- zsh history --
if [ -f "${HOME}/.zsh_history" ]; then
    cp -p "${HOME}/.zsh_history" "${STAGE}/zsh_history"
    echo "[stage] copied ~/.zsh_history"
else
    echo "[stage] no ~/.zsh_history on host, container starts with empty history"
fi

# --------------------------------------------------------------------- ssh --
if [ -d "${HOME}/.ssh" ]; then
    mkdir -p "${STAGE}/ssh"
    # Regular files only (-L follows symlinks, e.g. a config managed by a
    # dotfiles repo). This skips ControlMaster sockets, which cp cannot copy.
    count=0
    while IFS= read -r -d '' f; do
        rel="${f#./}"
        mkdir -p "${STAGE}/ssh/$(dirname "${rel}")"
        if cp -p "${HOME}/.ssh/${rel}" "${STAGE}/ssh/${rel}" 2>/dev/null; then
            count=$((count + 1))
        fi
    done < <(cd "${HOME}/.ssh" && find -L . -type f -print0 2>/dev/null)
    echo "[stage] copied ${count} file(s) from ~/.ssh"
else
    echo "[stage] no ~/.ssh on host, skipping"
fi

exit 0
