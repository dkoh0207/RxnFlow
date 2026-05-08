"""Back-compat shim. Logic now lives in rxnflow.cli.train.unidock_mogfn.

Prefer `rxnflow train unidock-mogfn ...` after `pip install -e .`.
"""

from rxnflow.cli.train.unidock_mogfn import unidock_mogfn

if __name__ == "__main__":
    unidock_mogfn()
