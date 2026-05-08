import click

from rxnflow import __version__
from rxnflow.cli.sample import sample
from rxnflow.cli.train import train


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="rxnflow")
def main() -> None:
    """RxnFlow — synthesis-oriented GFlowNet CLI."""


main.add_command(train)
main.add_command(sample)
