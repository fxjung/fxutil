Plotting reference
==================

.. py:module:: fxutil.plotting

:doc:`figures` walks through an export. The definitions here live in
``fxutil.plotting``. The package root also exports ``SaveFigure``, ``figax``,
``evf``, ``easy_prop_cycle``, ``pad_range``, ``LabelMaker``,
``force_log_ticks`` and ``get_safe_color_axes``.

Figure creation and export
--------------------------

.. autoclass:: SaveFigure
   :members: __call__, get_figure, figax, output_dir, register_contrast_color

   .. py:attribute:: styles

      Style stacks keyed by ``"dark"`` and ``"light"``. Append an rcParams
      dictionary to an instance's stack to customize that variant for
      export and preview.

.. py:data:: DEFAULT_FILETYPES

   Default export extensions, currently ``["png"]``. To choose formats
   for your own figures, pass ``filetypes`` to the saver.

.. autofunction:: figax

Domains and limits
------------------

.. py:data:: evf

   Callable instance of :class:`Evf`. Both forms below return the
   domain and the evaluated values, ready for ``ax.plot(*pair)``.

   >>> import numpy as np
   >>> from fxutil import evf
   >>> x, y = evf[0:1:3j, lambda x, power: x**power, {"power": 2}]
   >>> y.tolist()
   [0.0, 0.25, 1.0]
   >>> evf(x, np.square)[1].tolist()
   [0.0, 0.25, 1.0]

.. autoclass:: Evf

.. autofunction:: pad_range

.. autofunction:: set_aspect

Colors, ticks and legends
-------------------------

.. autofunction:: easy_prop_cycle

.. py:data:: solarized_colors

   Dictionary of Solarized base and accent names mapped to hex colors.
   ``easy_prop_cycle(ax, N=8, cmap="solarized")`` selects the eight accents.

.. autoclass:: TwoScaleLogNorm
   :members: __call__, inverse

.. autofunction:: get_safe_color_axes

.. autofunction:: force_log_ticks

.. autoclass:: LabelMaker

For example, split a subplot before drawing a heatmap and direct the
colorbar into its own axes:

.. code-block:: python

   from fxutil import figax, get_safe_color_axes

   fig, original = figax()
   ax, cax = get_safe_color_axes(original)
   image = ax.imshow([[0, 1], [2, 3]])
   fig.colorbar(image, cax=cax)
