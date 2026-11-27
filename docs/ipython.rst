Interactive imports and notebook figures
========================================

Importing ``fxutil`` in an IPython shell or notebook populates the user
namespace as if you had executed ``from fxutil.imports.general import *``.
This also happens on the first import of a submodule via the package.
Existing names such as ``np`` or ``display`` can be rebound. In a regular
Python process, use explicit imports such as ``from fxutil import SaveFigure``.

In either environment the package sets the root logger to ``WARNING`` and
adds a formatted stream handler. Applications with an existing logging
configuration may therefore see a changed threshold or duplicated messages.

The interactive namespace
-------------------------

.. py:module:: fxutil.imports

.. py:module:: fxutil.imports.general

The convenience module collects these names for interactive work:

* Scientific Python: ``np``, ``pd``, ``scipy``, ``h5py``, ``nx``,
  ``mpl``, ``plt``, ``sns``, ``cycler`` and ``make_axes_locatable``.
* Python and IPython: ``Path``, ``m`` (math), ``it`` (itertools), ``ft``
  (functools), ``op`` (operator), ``dataclasses``, ``random``, ``json``,
  ``re``, ``display``, ``get_ipython`` and ``register_cell_magic``.
* fxutil's top-level exports, plus :func:`~fxutil.common.minmax` and
  :func:`~fxutil.common.mmr`.

Importing this module sets pandas display limits to 500 rows, 500 columns
and a width of 1000. To access the same names from a module object, use
``from fxutil.imports import datascience``.

Saving a cell with %%s
----------------------

Create one saver in the notebook. For example:

.. code-block:: python

   import fxutil
   sf = SaveFigure("plots", presentation=True, use_styles=["light"],
                   interactive_mode=None)

Then run a separate cell with the figure name on the first line:

.. code-block:: text

   %%s dose response
   fig, ax = sf.figax()
   ax.plot([0, 1, 2], [0, 0.5, 0.75])
   ax.set(xlabel="Dose", ylabel="Response")

This writes ``plots/dose-response-light.png``. The magic searches the
user namespace for a ``SaveFigure`` instance regardless of variable name.
If none exists, it creates a default saver named ``sf``. More than one
matching binding raises ``RuntimeError``, even if two names refer to the
same object. The cell body executes in the user namespace on every drawing
pass, so assignments persist and other side effects repeat.

Session detection
-----------------

.. automodule:: fxutil.meta
   :members: in_ipython_session
