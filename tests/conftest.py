"""Shared pytest fixtures and configuration."""
import sys
from pathlib import Path

# Make repo root importable so tests can `import convert`, `import materials`, etc.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
