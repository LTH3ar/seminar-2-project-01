# Dev container image for AI4SE Project 1 - Issue Report Classification.
#
# Base: the official Docker Hub Python image, Debian 13 "trixie", slim variant.
# Multi-arch (amd64 and arm64/v8), so Intel and Apple Silicon members get the
# same environment. The slim variant starts at ~49 MB; everything the project
# needs is added explicitly below rather than inherited from a vendor image.
#
# Python is pinned to 3.11 because it is the newest version for which every
# library the four tracks need -- torch, transformers, setfit,
# sentence-transformers -- ships prebuilt wheels for both architectures.
# To move to 3.12 or 3.13, change PYTHON_VERSION and rebuild; nothing else in
# the project depends on the version.
#
# For a fully reproducible build, pin the digest instead of the tag:
#   FROM python:3.11-slim-trixie@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-trixie

# LaTeX adds roughly 1 GB to a 0.5 GB image. It is on by default because the
# course expects the report in LaTeX and this avoids a second toolchain, but
# members who only write code can skip it:
#   "build": { "args": { "INSTALL_LATEX": "false" } }   in devcontainer.json
ARG INSTALL_LATEX=true

# Non-root user. The official Python image has none, so it is created here.
# UID/GID 1000 matches the first human user on most Linux hosts, which keeps
# bind-mounted files writable from both sides.
# USERNAME is a non-root account name, not a credential.
# hadolint ignore=DL3064
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive \
    NLTK_DATA=/home/vscode/nltk_data

# ---------------------------------------------------------------------------
# System packages
#
#   ca-certificates      HTTPS for pip and the dataset download
#   git, openssh-client  the project is graded on its commit history
#   less, procps         required by the VS Code server that attaches later
#   libatomic1           likewise (VS Code server runtime dependency)
#   sudo                 devcontainer convention: passwordless sudo for the user
#   build-essential      source builds for any dependency without a wheel
#   graphviz             optional architecture diagrams for the report
#   curl                 used by post-create.sh and by GitHub CLI installs
# ---------------------------------------------------------------------------
# apt versions are deliberately unpinned: Debian publishes security updates
# within trixie, and pinning would break rebuilds every time it does. The
# Python toolchain below is pinned instead, since that is what affects results.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openssh-client \
        less \
        procps \
        libatomic1 \
        sudo \
        build-essential \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# LaTeX toolchain (optional, see INSTALL_LATEX above)
# ---------------------------------------------------------------------------
# hadolint ignore=DL3008
RUN if [ "${INSTALL_LATEX}" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            latexmk \
            texlive-latex-recommended \
            texlive-latex-extra \
            texlive-fonts-recommended \
            texlive-bibtex-extra \
            biber \
        && rm -rf /var/lib/apt/lists/* ; \
    else \
        echo "Skipping LaTeX toolchain (INSTALL_LATEX=${INSTALL_LATEX})" ; \
    fi

# ---------------------------------------------------------------------------
# Non-root user
# ---------------------------------------------------------------------------
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# ---------------------------------------------------------------------------
# Python dependencies
#
# Kept in its own layer, before any source is copied, so that editing code does
# not invalidate this (slow) layer on rebuild.
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade \
        pip==26.2.1 \
        setuptools==84.0.0 \
        wheel==0.48.0 \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ---------------------------------------------------------------------------
# NLTK corpora used by the cleaning pipeline.
# Baked into the image rather than fetched at container start, so the notebook
# runs offline.
# ---------------------------------------------------------------------------
RUN mkdir -p ${NLTK_DATA} \
    && python -m nltk.downloader -d ${NLTK_DATA} stopwords wordnet omw-1.4 punkt

# Cache directories are created here, and owned by the user, so that the named
# volumes mounted over them in devcontainer.json inherit the right ownership.
RUN mkdir -p /home/${USERNAME}/.cache/pip \
             /home/${USERNAME}/.cache/huggingface \
             /home/${USERNAME}/.vscode-server \
    && chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}

WORKDIR /workspaces/ai4se-project
USER ${USERNAME}

# Anything the user pip-installs at runtime lands here.
ENV PATH=/home/vscode/.local/bin:${PATH}

CMD ["sleep", "infinity"]
