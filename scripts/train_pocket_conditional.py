"""Back-compat shim. Logic now lives in rxnflow.cli.train.pocket.

Prefer `rxnflow train pocket ...` after `pip install -e .`.
"""

from rxnflow.cli.train.pocket import pocket

if __name__ == "__main__":
    pocket()
