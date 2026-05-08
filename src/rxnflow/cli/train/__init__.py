import importlib

import click


_COMMANDS = {
    "unidock": "rxnflow.cli.train.unidock:unidock",
    "unidock-moo": "rxnflow.cli.train.unidock_moo:unidock_moo",
    "unidock-mogfn": "rxnflow.cli.train.unidock_mogfn:unidock_mogfn",
    "seh": "rxnflow.cli.train.seh:seh",
    "pretrain-qed": "rxnflow.cli.train.pretrain_qed:pretrain_qed",
    "pocket": "rxnflow.cli.train.pocket:pocket",
    "few-shot-unidock": "rxnflow.cli.train.few_shot_unidock:few_shot_unidock",
}


class _LazyTrainGroup(click.Group):
    def list_commands(self, ctx):
        return list(_COMMANDS)

    def get_command(self, ctx, name):
        target = _COMMANDS.get(name)
        if target is None:
            return None
        module_path, attr = target.split(":")
        return getattr(importlib.import_module(module_path), attr)


@click.group(cls=_LazyTrainGroup)
def train() -> None:
    """Run training / optimization."""
