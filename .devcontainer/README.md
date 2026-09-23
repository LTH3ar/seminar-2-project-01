# Dev container

> **Not the supported setup path.** A plain virtualenv is — see `RUNBOOK.md`.
> This container is provided as-is for anyone who prefers it; if it gives you
> trouble, use the virtualenv instead rather than debugging the container.

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
