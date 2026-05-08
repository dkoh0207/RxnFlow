"""Back-compat shim. Logic now lives in rxnflow.cli.train.pretrain_qed.

Prefer `rxnflow train pretrain-qed ...` after `pip install -e .`.
"""

from rxnflow.cli.train.pretrain_qed import pretrain_qed

if __name__ == "__main__":
    pretrain_qed()
