"""Back-compat shim. Logic now lives in rxnflow.cli.sample.unidock.

Prefer `rxnflow sample unidock ...` after `pip install -e .`.
"""

from rxnflow.cli.sample.unidock import unidock

if __name__ == "__main__":
    unidock()
