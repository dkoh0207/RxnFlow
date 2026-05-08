"""Back-compat shim. Logic now lives in rxnflow.cli.train.unidock.

Prefer `rxnflow train unidock ...` after `pip install -e .`.
"""

from rxnflow.cli.train.unidock import unidock

if __name__ == "__main__":
    unidock()
