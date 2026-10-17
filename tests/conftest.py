"""Make the analysis package importable for the test suite.

The analysis scripts are also runnable directly (`python3 analysis/p1_exact_power.py`),
where the script's own directory is already on sys.path.  Under pytest the root is
the repository, so add `analysis/` explicitly.
"""
import pathlib
import sys

ANALYSIS = pathlib.Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
