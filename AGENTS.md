# Working on fxutil

## Repository map

- The installable package lives in `src/fxutil/`. Check `__init__.py` for root re-exports and import-time logging and IPython behavior. `imports/general.py` defines the interactive namespace and the `%%s` cell magic.
- `plotting.py` implements `SaveFigure`, figure sizing and axes helpers. The six Matplotlib style sheets in `mplstyles/` ship as package data. The publication, presentation, light and dark settings compose in the order specified by the saver.
- `common.py` contains formatting, collection and repository helpers. `typing.py` implements the runtime `Combi` decorator. Its annotation inspection matters to both plotting and user-defined functions.
- `cli.py` maps Typer arguments to the Python function in `manuscript.py`. Inspect both when changing an option.
- `test/` contains the pytest suite and a runnable presentation gallery. The user guide and reference live in `docs/`. Reference pages pull signatures and docstrings from the installed package.

## Development and checks

Work from the repository root. `pyproject.toml` specifies Python 3.13 or later. Use `uv sync --dev` for the environment used by CI. The Sphinx tools are in the dev **dependency group**. The optional `dev` extra has different contents.

```bash
MPLBACKEND=Agg uv run pytest
uv run sphinx-build -E -W -b html docs docs/_build/html
uv run sphinx-build -W -b doctest docs docs/_build/doctest
```

Use `uv run pre-commit install` to enable the configured Ruff hooks. To review the appearance of presentation changes, generate the gallery and inspect the light and dark single- and multi-panel PNGs and PDFs:

```bash
MPLBACKEND=Agg uv run python test/test_presentation_rendering.py /tmp/fxutil-preview
```

The Agg backend avoids GUI display. Publication rendering still needs LaTeX and the packages listed in `mplstyles/tex.mplstyle`. The test workflow installs TeX Live. For changes to manuscript packaging, exercise the CLI against a disposable source tree containing a real figure, table and bibliography. Check the rewritten TeX, rebuilt PDF and archive contents. The default command removes earlier packaged outputs before validating the source.

## Keeping the public interface documented

Update the relevant Sphinx guide and the defining docstrings when behavior changes. Check the root and interactive re-exports when adding or renaming public helpers. Keep the README examples short enough to run from a fresh installation. Build the HTML from a clean Sphinx environment and follow its navigation from `docs/_build/html/index.html`.

For plotting changes, check saved and preview styles separately. The callback may run multiple times, and the two paths handle text and figure lifetimes separately. For CLI changes, check `uv run fxutil manuscript package --help` against the Python entry point and the manuscript guide. Record observed limitations in the guide and ask the maintainer when an explanation depends on unknown design intent.

Versions come from Git tags via setuptools-scm. The workflows in `.github/workflows/` define testing on pushes and pull requests to master, publication on a published release, and Pages deployment after publication.
