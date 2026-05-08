"""Back-compat shim. Logic now lives in rxnflow.cli.train.unidock_moo.

Prefer `rxnflow train unidock-moo ...` after `pip install -e .`.
"""

from rxnflow.cli.train.unidock_moo import unidock_moo

if __name__ == "__main__":
    unidock_moo()
