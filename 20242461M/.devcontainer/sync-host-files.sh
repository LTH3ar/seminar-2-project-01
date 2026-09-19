#!/usr/bin/env bash
#
# Runs INSIDE the container on every start (devcontainer.json ->
# postStartCommand).
#
# Takes the read-only copies staged by stage-host-files.sh (mounted at
# ~/.host-seed) and installs them with the ownership and permissions ssh and
# zsh expect. Works on copies of copies: the host originals are not mounted.
#
#   history   seeded into the persistent volume ONCE, when it is still empty.
#             After that the container keeps its own history across rebuilds
#             and never writes back to the host.
#   ssh       refreshed on every start, so a key added on the host shows up
#             after a container restart. known_hosts is merged, not replaced,
#             so hosts accepted inside the container are kept.

set -uo pipefail

SEED="${HOME}/.host-seed"
STATE="${HOME}/.zsh_state"

if [ ! -d "${SEED}" ]; then
    echo "[sync] ${SEED} not mounted, nothing to import"
    exit 0
fi

host_home="$(cat "${SEED}/host_home" 2>/dev/null || true)"

# ------------------------------------------------------------- zsh history --
mkdir -p "${STATE}"
if [ -s "${SEED}/zsh_history" ] && [ ! -s "${STATE}/zsh_history" ]; then
    cp "${SEED}/zsh_history" "${STATE}/zsh_history"
    chmod 600 "${STATE}/zsh_history"
    echo "[sync] seeded zsh history ($(wc -l < "${STATE}/zsh_history") lines)"
fi

# --------------------------------------------------------------------- ssh --
if [ -d "${SEED}/ssh" ]; then
    install -d -m 700 "${HOME}/.ssh"

    while IFS= read -r -d '' f; do
        rel="${f#./}"
        [ "${rel}" = "known_hosts" ] && continue
        install -D -m 600 "${SEED}/ssh/${rel}" "${HOME}/.ssh/${rel}"
    done < <(cd "${SEED}/ssh" && find . -type f -print0)

    find "${HOME}/.ssh" -type f -name '*.pub' -exec chmod 644 {} +

    # known_hosts: host entries first, then anything accepted in the
    # container, duplicates dropped.
    if [ -f "${SEED}/ssh/known_hosts" ]; then
        tmp="$(mktemp)"
        cat "${SEED}/ssh/known_hosts" "${HOME}/.ssh/known_hosts" 2>/dev/null \
            | awk 'NF && !seen[$0]++' > "${tmp}"
        install -m 644 "${tmp}" "${HOME}/.ssh/known_hosts"
        rm -f "${tmp}"
    fi

    # config: make a macOS/host config usable on Linux.
    cfg="${HOME}/.ssh/config"
    if [ -f "${cfg}" ]; then
        tmp="$(mktemp)"
        {
            # UseKeychain is Apple-only; Linux ssh aborts on it without this.
            # Only added if the user has no IgnoreUnknown of their own, since
            # the first IgnoreUnknown in the file wins.
            if ! grep -qiE '^[[:space:]]*IgnoreUnknown' "${cfg}"; then
                echo "IgnoreUnknown UseKeychain"
            fi
            # Rewrite absolute host paths (/Users/alice/.ssh/...) to ours.
            if [ -n "${host_home}" ] && [ "${host_home}" != "${HOME}" ]; then
                escaped="$(printf '%s' "${host_home}" | sed 's/[][\.*^$/|]/\\&/g')"
                sed "s|${escaped}|${HOME}|g" "${cfg}"
            else
                cat "${cfg}"
            fi
        } > "${tmp}"
        install -m 600 "${tmp}" "${cfg}"
        rm -f "${tmp}"
    fi

    echo "[sync] ssh files installed in ~/.ssh"
fi