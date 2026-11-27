Packaging a LaTeX manuscript
============================

.. py:module:: fxutil.cli

``fxutil manuscript package`` produces an editable submission alongside the
original source directory. It renames figures, inlines tables and a compiled
bibliography, rebuilds the PDF, and archives the resulting directory. Run it
on a manuscript that already builds with ``latexmk -pdf``. The command needs
``latexmk``, the TeX tools and packages used by the manuscript, and the
``zip`` executable on your path.

Source layout
-------------

For example, start with this directory::

   paper/
     manuscript.tex
     figures/decay.pdf
     tables/rates.tex
     references.bib

Use literal commands like these in the main file:

.. code-block:: latex

   \documentclass{article}
   \usepackage{graphicx}
   \begin{document}
   \includegraphics{figures/decay.pdf}
   \input{tables/rates}
   See \cite{example}.
   \bibliographystyle{plain}
   \bibliography{references}
   \end{document}

With a table fragment in ``rates.tex`` and a BibTeX entry named ``example``,
run the following from the parent of ``paper``:

.. code-block:: console

   $ fxutil manuscript package paper --no-delete-existing

This creates ``paper-packaged/`` and ``paper-packaged.zip`` beside
``paper/``. The directory contains the rewritten ``manuscript.tex``,
``manuscript.pdf`` and ``figures/figure-1.pdf``. Table text and the contents
of the generated ``.bbl`` are embedded in the main file. Top-level
``.bst`` files are copied for compilation. The packager removes the
``.bib``, ``.bbl`` and ``.bst`` build inputs from the output after rebuilding.

Options and replacement
-----------------------

=============================  ===============================================================
Argument or option             Meaning
=============================  ===============================================================
``SRC_DIR``                    Source directory. Relative paths in the main file start here.
``--tex-name``                 Main file in the source root. Default ``manuscript.tex``.
``--tables-dir``               Directory prefix of table inputs. Default ``tables``.
``--figures-dest-dir``         Output figure directory name. Default ``figures``.
``--figures-dir``              Accepted (default ``figures``). Source lookup uses each literal path.
``--delete-existing``          Default. Remove the prior directory and ZIP before checking input.
``--no-delete-existing``       Skip deletion. An existing output directory raises ``FileExistsError``.
=============================  ===============================================================

Choose a fresh destination by working on a differently named source copy if
you need to keep a previous package. With deletion disabled, an existing
ZIP alone can still be updated by ``zip``.

What the packager recognizes
----------------------------

* Include at least one literal ``\includegraphics{path.ext}`` in the main
  file. A manuscript with no matching figure currently fails during path
  substitution. Give the full filename and extension, one command per line,
  with no optional arguments or trailing brace groups on that line. Paths
  are replaced throughout the main text. Repeated paths use the number of
  their last occurrence.
* ``\input{tables/rates}`` reads ``tables/rates.tex`` once. Include the
  directory and omit the extension. Figure discovery runs before table
  insertion. Put figure references in the main file.
* A bibliography is optional. Use at most one literal
  ``\bibliography{basename}`` naming one ``.bib`` file without its extension.
  The command expects BibTeX's ``.bbl`` output.
* Matching uses regular expressions on the source. It does not follow
  arbitrary inputs, expand macros, or collect local classes, packages or
  other assets. Figure matching also sees commented lines. A later pass
  strips matched percent comments except ones containing ``ORCID``.

Inspect the rebuilt PDF and the ZIP member paths before submission. The
archiver receives the directory's absolute path, so archive members can
include parent directory names. The packager checks for expected build
files but does not check subprocess return codes. Its final “Exported”
message alone does not establish that a ZIP was written successfully.

Python entry point
------------------

.. automodule:: fxutil.manuscript
   :members: package_manuscript
