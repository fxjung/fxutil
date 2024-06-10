from typing import Annotated

import typer

from pathlib import Path

from fxutil.manuscript import package_manuscript as _package_manuscript

import logging

log = logging.getLogger(__name__)

app = typer.Typer()
manuscript_app = typer.Typer()

app.add_typer(
    manuscript_app, name="manuscript", help="Commands for working with manuscripts"
)


@manuscript_app.command("package")
def package_manuscript(
    src_dir: Annotated[
        str, typer.Argument(help="The directory containing the manuscript source")
    ],
    tex_name: Annotated[
        str,
        typer.Option(help="The name of the manuscript's main TeX file"),
    ] = "manuscript.tex",
    figures_dir: Annotated[
        str,
        typer.Option(
            help="The name of the directory containing the manuscript's figures",
        ),
    ] = "figures",
    tables_dir: Annotated[
        str,
        typer.Option(
            help="The name of the directory containing the manuscript's tables",
        ),
    ] = "tables",
    figures_dest_dir: Annotated[
        str,
        typer.Option(
            help="The name of the directory containing the manuscript's figures in the packaged version",
        ),
    ] = "figures",
    delete_existing: Annotated[
        bool,
        typer.Option(
            help="Whether to delete any existing packaged directory and zip file",
        ),
    ] = True,
):
    _package_manuscript(
        submission_src_dir=src_dir,
        tex_name=tex_name,
        figures_src_dir_name=figures_dir,
        tables_src_dir_name=tables_dir,
        figures_dest_dir_name=figures_dest_dir,
        delete_existing=delete_existing,
    )
