# conftest.py — top-level pytest configuration
#
# Ensures the project root is on sys.path so that `import sakura` and
# `import agents` work correctly when running `pytest` from any directory.

import sys
from pathlib import Path

# Add the project root to the front of sys.path
sys.path.insert(0, str(Path(__file__).parent))
