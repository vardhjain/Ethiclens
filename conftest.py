"""Root pytest configuration.

Puts the repository root on ``sys.path`` so the flat ``ml`` package (dataset
loaders, the golden oracle) is importable in tests. ``fairness_core`` is an
installed (editable) package and needs no such help.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
