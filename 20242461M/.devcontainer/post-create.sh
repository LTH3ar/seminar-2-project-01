#!/usr/bin/env bash
#
# Runs once, after the dev container is created.
#
#   1. installs the project in editable mode so `import ai4se` works from
#      anywhere without sys.path manipulation;
#   2. verifies the NLTK corpora baked into the image are reachable;
#   3. caches the NLBSE'24 dataset into data/raw/;
#   4. runs the test suite as a smoke check.
#
# Every step is non-fatal except the editable install: a member without
# network access at first launch should still land in a usable container.

set -uo pipefail

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
step() { echo -e "\n${BOLD}==> $*${RESET}"; }
warn() { echo -e "${YELLOW}    ! $*${RESET}"; }
ok()   { echo -e "${GREEN}    ok${RESET} $*"; }

cd "$(dirname "$0")/.." || exit 1

# ---------------------------------------------------------------- 1. install --
step "Installing ai4se in editable mode"
if pip install --no-cache-dir -e ".[dev]"; then
    ok "import ai4se now works from any directory"
else
    echo "    FAILED - the container is not usable, check pyproject.toml" >&2
    exit 1
fi

# ------------------------------------------------------------------ 2. nltk --
step "Checking NLTK corpora"
python - <<'PY'
import nltk

missing = []
for resource, path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
]:
    try:
        nltk.data.find(path)
        print(f"    ok {resource}")
    except LookupError:
        missing.append(resource)

for resource in missing:
    print(f"    downloading {resource} ...")
    nltk.download(resource, quiet=True)
PY

# --------------------------------------------------------------- 3. dataset --
step "Caching the NLBSE'24 dataset"
if python -c "
from ai4se.loader import load_dataset
data = load_dataset(kind='memory')
for split, repository in data.items():
    print(f'    {split:<6} {len(repository):>5} issues, '
          f'{len(repository.repos())} projects, '
          f'{repository.label_distribution()}')
"; then
    ok "data/raw/ populated"
else
    warn "download failed (offline?). Run 'make data' once you have network."
fi

# ----------------------------------------------------------------- 4. tests --
step "Running the test suite"
if python -m pytest tests/ -q; then
    ok "environment verified"
else
    warn "tests failed - inspect with 'python -m pytest tests/ -v'"
fi

cat <<'EOF'

────────────────────────────────────────────────────────────────────────
  AI4SE Project 1 — environment ready

  make eda        run the Track A notebook end to end
  make lab        start Jupyter Lab on port 8888
  make test       run the test suite
  make check      verify the persistence-layer equivalence (prints the
                  table that goes into the report)
  make report     compile report/report.tex with latexmk

  Import the package directly — no sys.path juggling needed:

      from ai4se.loader import load_split
      train = load_split("train", kind="memory")
────────────────────────────────────────────────────────────────────────
EOF

sudo chsh -s $(which zsh)
echo "$(which zsh)" >> ~/.bashrc