# Dev container image for AI4SE Project 1 - Issue Report Classification.
#
# Python is pinned to 3.11: it is the newest version for which every library
# the four tracks need (torch, transformers, setfit, sentence-transformers)
# ships prebuilt wheels for both linux/amd64 and linux/arm64. Pinning also
# means every group member and the lecturer run byte-identical environments,
# which is the point of committing a dev container at all.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/home/vscode/nltk_data

# System packages:
#   build-essential  - source builds for any wheel-less dependency
#   graphviz         - optional model/architecture diagrams for the report
#   latexmk + texlive-* - so the LaTeX report can be compiled in-container
#                      (the course strongly advises LaTeX as thesis practice)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        graphviz \
        latexmk \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-bibtex-extra \
        biber \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies as a separate layer so that editing source code
# does not invalidate the (slow) dependency layer on rebuild.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# NLTK corpora used by the cleaning pipeline. Baked into the image rather than
# downloaded at container start, so the notebook runs offline.
RUN mkdir -p ${NLTK_DATA} \
    && python -m nltk.downloader -d ${NLTK_DATA} stopwords wordnet omw-1.4 punkt \
    && chown -R vscode:vscode ${NLTK_DATA}

WORKDIR /workspaces/ai4se-project
