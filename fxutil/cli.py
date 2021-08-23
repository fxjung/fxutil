import click

from pathlib import Path
from file_helpers import estimate_progress as _estimate_progress


@click.group()
def cli():
    """fxutil cli interface"""
    ...


@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
)
def estimate_progress(directory):
    """
    TODO
    """
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory).expanduser()

    _estimate_progress(directory)


if __name__ == "__main__":
    cli()
