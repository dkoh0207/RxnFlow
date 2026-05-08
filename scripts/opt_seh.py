"""Back-compat shim. Logic now lives in rxnflow.cli.train.seh.

Prefer `rxnflow train seh ...` after `pip install -e .`.
"""

from rxnflow.cli.train.seh import seh

if __name__ == "__main__":
    seh()
