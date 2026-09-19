# Dev container

Built from the official Docker Hub image `python:3.11-slim-trixie`
(Debian 13 "trixie", slim variant, multi-arch amd64 + arm64/v8).

No vendor devcontainer image and no `features` block: the non-root user, git,
sudo, and the VS Code server prerequisites are all installed explicitly in the
Dockerfile, so the only upstream layer is the official Python image.

| File | Purpose |
|---|---|
| `Dockerfile` | Image definition. Build context is the repository root. |
| `devcontainer.json` | Editor settings, extensions, port forwarding, volumes. |
| `post-create.sh` | Runs once after creation: editable install, NLTK check, dataset cache, test smoke check. |
| `stage-host-files.sh` | Runs **on the host** before each start: copies `~/.zsh_history` and `~/.ssh` into a private staging dir. |
| `sync-host-files.sh` | Runs in the container on each start: installs the staged copies with correct permissions. |
| `zshrc` | The container user's `~/.zshrc`: history, completion, git prompt, autosuggestions, fzf. |

## Shell, history and SSH keys

The `vscode` user's shell is zsh, with menu completion, a git-branch prompt,
prefix history search on Up/Down, inline suggestions (Right arrow accepts),
syntax highlighting, and fzf on Ctrl-R / Ctrl-T.

Your host's history and SSH keys are brought in **as copies**; the originals
are only read, never mounted and never written:

```
host ~/.zsh_history, ~/.ssh/*          (read only by cp)
        │  stage-host-files.sh  (initializeCommand, on the host)
        ▼
host ~/.cache/ai4se-devcontainer/host-seed/     (private copy, outside the repo)
        │  bind mount, readonly
        ▼
container ~/.host-seed/
        │  sync-host-files.sh  (postStartCommand)
        ▼
container ~/.ssh/                      refreshed every start
container ~/.zsh_state/zsh_history     seeded once, then kept in a volume
```

- **History** is imported once, into the `ai4se-zsh-state` volume, and then
  lives on its own: container commands never reach the host file, and survive
  rebuilds. To re-import the host history, run
  `docker volume rm ai4se-zsh-state` and rebuild.
- **SSH** files are re-copied at every container start, so a new host key
  appears after a restart. `known_hosts` is merged, so hosts accepted inside
  the container are kept. A macOS `config` is made Linux-safe
  (`IgnoreUnknown UseKeychain`, `/Users/you/...` paths rewritten).
- If the host runs an ssh-agent, VS Code forwards it and that is used first.
  Otherwise the container starts its own; run `ssh-add` once per start to
  enter a passphrase a single time.
- Git identity (`user.name`, `user.email`) is copied from the host's
  `~/.gitconfig` automatically by VS Code.
- Windows hosts: open the repository from WSL. `initializeCommand` needs bash
  and the mount uses `$HOME`.

## Build arguments

| Argument | Default | Effect |
|---|---|---|
| `PYTHON_VERSION` | `3.11` | Newest version with prebuilt wheels for torch, transformers and setfit on both architectures. |
| `INSTALL_LATEX` | `true` | `latexmk` + TeX Live, roughly 1 GB. Set `false` if you are not compiling the report. |
| `USER_UID` / `USER_GID` | `1000` | Match your host user if bind-mounted files come out root-owned. |

Change them under `build.args` in `devcontainer.json`, then rebuild the
container.

## Reproducible builds

The tag `python:3.11-slim-trixie` moves as Debian publishes updates. For a
build that is identical months from now, pin the digest in the Dockerfile:

```dockerfile
FROM python:3.11-slim-trixie@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534
```

## Checks

```bash
hadolint .devcontainer/Dockerfile          # passes clean
devcontainer build --workspace-folder .    # devcontainer CLI
```
