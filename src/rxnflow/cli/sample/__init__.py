import importlib

import click


_COMMANDS = {
    "unidock": "rxnflow.cli.sample.unidock:unidock",
    "zero-shot": "rxnflow.cli.sample.zero_shot:zero_shot",
}


class _LazySampleGroup(click.Group):
    def list_commands(self, ctx):
        return list(_COMMANDS)

    def get_command(self, ctx, name):
        target = _COMMANDS.get(name)
        if target is None:
            return None
        module_path, attr = target.split(":")
        return getattr(importlib.import_module(module_path), attr)


@click.group(cls=_LazySampleGroup)
def sample() -> None:
    """Sample from trained models."""
