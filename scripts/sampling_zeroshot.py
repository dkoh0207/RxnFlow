"""Back-compat shim. Logic now lives in rxnflow.cli.sample.zero_shot.

Prefer `rxnflow sample zero-shot ...` after `pip install -e .`.
"""

from rxnflow.cli.sample.zero_shot import zero_shot

if __name__ == "__main__":
    zero_shot()
