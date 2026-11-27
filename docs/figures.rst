Saving figures
==============

Put the drawing code in a zero-argument function that creates a fresh pyplot
figure each time it runs. :class:`~fxutil.plotting.SaveFigure` calls it once
per saved style. Each pass writes all requested formats and closes the figure.
An enabled preview runs the function once more. Compute data and draw random
samples **before** the callback so every version shows the same observations.

.. code-block:: python

   import numpy as np
   from fxutil import SaveFigure

   t = np.linspace(0, 4, 101)
   response = np.exp(-t)
   sf = SaveFigure("plots", filetypes=["pdf", "png"], interactive_mode=None)

   def draw():
       fig, ax = sf.figax()
       ax.plot(t, response)
       ax.axhline(0.5, color="contrast", linestyle="--")
       ax.set(xlabel=r"Time $t$ (s)", ylabel=r"Response $R(t)$")

   sf(draw, "decay curve")

This publication example writes ``decay-curve-dark.pdf``,
``decay-curve-light.pdf`` and their PNG counterparts in ``plots/``.
It requires a working Matplotlib LaTeX installation, including the
``commath``, ``amsfonts`` and ``lmodern`` packages used by the style.

Files and previews
------------------

* By default both styles are saved as PNG at 250 dpi. Use
  ``use_styles=["light"]`` for one style. Pass ``filetypes="pdf"`` on a
  single call to override the constructor setting.
* The default output location in a regular Git checkout is
  ``<repository>/data/figures``. Outside a repository it is ``./figures``.
  Inspect ``sf.output_dir`` to see the base path. Use
  ``subfolder_per_filetype=True`` for paths such as ``plots/pdf/decay-curve-light.pdf``.
  Reusing a name with the same style and extension overwrites that file.
* The default preview uses the dark style at 96 dpi. Set ``interactive_mode``
  independently to ``"light"``, ``"dark"`` or ``None``. An IPython preview
  uses ``display(fig)``. A script uses ``plt.show()``.
* Export replaces underscores with spaces in axis labels, axes titles,
  legend entries and legend titles unless the string contains ``$``.
  To retain literal underscores set ``make_tex_safe=False``. This
  substitution runs only on saved versions and covers only underscores.

Papers and slides
-----------------

.. py:module:: fxutil.mplstyles

===================  ==========================  =====================================
Setting              Publication (default)       Presentation (``presentation=True``)
===================  ==========================  =====================================
Single-panel canvas  170 × 127.5 mm              270 × 202.5 mm
Text                 10 pt serif, LaTeX          26 pt sans serif, 30 pt titles/labels
Mathematics          LaTeX                       Native Computer Modern mathtext
Export bounds        Nominal canvas by default   Tight bounds plus 0.08 inch padding
===================  ==========================  =====================================

Presentation uses Atkinson Hyperlegible Next when installed, with DejaVu Sans
as fallback. Curves are 3.2 pt wide and markers are 9 pt. Left-aligned titles,
outward ticks and frameless legends leave room for the data. Use
``ax.grid(True)`` for a grid behind the curves. Size the source canvas for its
final placement on the slide. Shrinking the exported image also shrinks its
text and strokes.

The default export has a transparent background. Place a dark export on a
dark surface. For an opaque black slide figure, set **both** the transparency
flag and the face color. The bundled dark style sets a transparent figure
face even when transparency is disabled:

.. code-block:: python

   slides = SaveFigure(presentation=True, use_styles=["dark"],
                       interactive_mode=None, output_transparency=False)
   slides.styles["dark"].append({"figure.facecolor": "black"})

Each instance has its own style lists. You can also set Matplotlib rcParams
inside the drawing function before creating the figure. For full LaTeX on
slides, set ``text.usetex`` to ``True`` and ``font.sans-serif`` to
``["Computer Modern Sans Serif"]`` there. The publication preamble is
available in presentation mode. Within a callback the named color
``"contrast"`` is black on light and white on dark. ``"acontrast"`` is the
opposite. These two names are registered globally in Matplotlib.

To use the same settings on a figure you manage yourself, pass a saver's
style stack to ``plt.style.context(sf.styles["light"], after_reset=True)``.
The stacks combine Matplotlib's base style with the sheets shipped in
``fxutil.mplstyles``. If the drawing code uses the named contrast colors,
call ``sf.register_contrast_color("light")`` first.

Panels and dimensions
---------------------

Use ``sf.figax()`` for one axes. Give grid dimensions to get a flat,
row-major list when the grid has several cells:

.. code-block:: python

   def draw_panels():
       fig, (signal, rate) = sf.figax(
           n_rows=2, n_cols=1, height_ratios=[3, 1], width=150, height=110
       )
       signal.plot(t, response)
       rate.plot(t, np.gradient(response, t))
       signal.set_ylabel("Response")
       rate.set(xlabel="Time (s)", ylabel="Rate (1/s)")

   sf(draw_panels, "decay rate")

Specifying only ``n_panels`` selects two columns and enough whole rows to
fit the count. Three requested panels yield four axes. Specify ``n_cols=1``
for a single column. Multi-cell grids get letter labels in TeX markup.
For native presentation text pass ``panel_labels=[]`` and add any letters
with ``ax.text``. In TeX mode, ``panel_labels=[0, 2]`` labels only the
first and third cells.

Dimensions on the saver and its methods are in millimetres. Per-call
``width`` and ``height`` each override their constructor defaults. An explicit
height fixes the total canvas height. Otherwise the height is
``width / aspect_ratio * n_rows / n_cols``. Use ``sf.get_figure(rows, cols)``
for an empty figure of the same size. The standalone
:func:`fxutil.plotting.figax` takes a Matplotlib size in inches.

The saver applies constrained layout after drawing. To pass settings to the
engine, use e.g. ``sf(draw, "decay", layout_engine_params={"w_pad": 0.1})``
(padding in inches for this engine). To apply the rectangle, padding and
spacing supplied to the constructor, pass ``layout_engine_params=sf.layout_engine_params``.
For manually positioned axes use ``layout_engine="none"``.

See :doc:`plotting-api` for all call signatures and the axes helpers.
