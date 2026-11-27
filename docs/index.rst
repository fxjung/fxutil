fxutil
======

.. py:module:: fxutil

fxutil is a Python toolkit for scientific work. Its figure saver renders one
Matplotlib plotting function for a paper or a slide deck. Its manuscript
command turns a LaTeX source directory into a submission archive. The
smaller helpers handle recurring tasks such as formatting an axis label,
checking that a group contains one distinct value, or setting up a notebook.

Install with Python 3.13 or later::

   python -m pip install fxutil

To save a first figure without a GUI or a TeX installation:

.. code-block:: python

   from fxutil import SaveFigure, evf

   sf = SaveFigure("figures", presentation=True, use_styles=["light"],
                   interactive_mode=None)

   def draw():
       fig, ax = sf.figax()
       ax.plot(*evf[0:3:100j, lambda t: t**2])
       ax.set(xlabel="Time (s)", ylabel="Squared time (s²)")

   sf(draw, "time squared")

The result is ``figures/time-squared-light.png``. See :doc:`figures` for
publication typography, multiple formats, panel sizing and previews.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   figures
   manuscript
   utilities
   ipython
   plotting-api

:ref:`genindex` · :ref:`modindex` · :ref:`search`
