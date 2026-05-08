"""Back-compat shim. Logic now lives in rxnflow.cli.train.few_shot_unidock.

Prefer `rxnflow train few-shot-unidock ...` after `pip install -e .`.
"""

from rxnflow.cli.train.few_shot_unidock import few_shot_unidock

if __name__ == "__main__":
    few_shot_unidock()
