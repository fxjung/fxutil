Working with values and collections
===================================

.. py:module:: fxutil.common

All helpers below are available from :mod:`fxutil.common`. The package
root also exports them, except for :func:`minmax` and :func:`mmr`, which are
available from this module and the :doc:`interactive imports <ipython>`.

Formatting and rounding
-----------------------

:func:`scinum` produces a fragment for use inside a LaTeX math expression.
Choose an empty suffix when no unit follows the number:

>>> from fxutil import scinum
>>> print(scinum(34_000_000, suffix=""))
3.4\times 10^{7}
>>> print(scinum(0.0025, force_mode="e", force_pref=True, suffix=""))
+2.5\times 10^{-3}
>>> print(scinum(34_000, force_mode="f", no_trailing_zeros=False))
34\,000.00\,

.. autofunction:: scinum

.. autofunction:: round_by_method

.. autofunction:: fmt_bytes

.. autofunction:: described_size

Selecting and checking values
-----------------------------

Choose the check that matches what you know about the input. For instance,
a sequence of repeated measurements can contain several entries yet still
have a single common value.

>>> from fxutil import bunny, nixt, onixt, permute
>>> nixt([5, 7])
5
>>> onixt([5])
5
>>> bunny([5, 5, 5])
5
>>> permute(["low", "medium", "high"], [2, 0, 2])
['high', 'low', 'high']

.. autofunction:: nixt

.. autofunction:: onixt

.. autofunction:: get_unique_with_bang

.. py:function:: bunny(iterable)

   Alias for :func:`get_unique_with_bang`.

.. autofunction:: permute

.. autofunction:: minmax

.. autofunction:: mmr

Example data and repository paths
---------------------------------

>>> from fxutil import thing
>>> thing("df").head(2).to_dict("list")
{'a': [10, 11], 'b': [100, 101]}
>>> from fxutil.common import mmr
>>> mmr(thing("list"))
(10, 19, 9)

.. autofunction:: thing

.. autofunction:: get_git_repo_path

Accepting a single value or an iterable
---------------------------------------

.. py:module:: fxutil.typing

Annotate a parameter with :data:`Combi` and decorate the function with
:func:`parse_combi_args` to handle calls such as ``formats("pdf")`` and
``formats(["png", "pdf"])`` in the same loop:

>>> from fxutil import Combi, parse_combi_args
>>> @parse_combi_args
... def formats(names: Combi[str] = "png"):
...     return [name.upper() for name in names]
>>> formats()
['PNG']
>>> formats("pdf")
['PDF']
>>> formats(["png", "pdf"])
['PNG', 'PDF']

.. py:data:: Combi

   Generic annotated union of a single item, an iterable of items, or a
   string. The annotation marks a parameter for the decorator. It can be
   used as ``Combi[T] | None`` for an optional argument.

.. autofunction:: parse_combi_args

.. autofunction:: parse_combi_argument

.. autofunction:: is_combi_ann

File helpers
------------

.. py:module:: fxutil.file_helpers

.. py:function:: estimate_progress(directory)

   Placeholder. Calling it returns ``None``.
