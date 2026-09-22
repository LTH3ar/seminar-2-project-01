"""pytest configuration for the 252276M test suite.

Puts src/ on sys.path so `import ai4se` works without installing the package.
"""
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
