import matplotlib.pyplot as plt
from pathlib import Path


class SaveFigure:
    def __init__(self, plot_dir, output_dpi=250, output_transparency=True):
        plot_dir = Path(plot_dir)
        self.plot_dirs = {}
        for ext in ["pdf", "png"]:
            self.plot_dirs[ext] = plot_dir / ext
            self.plot_dirs[ext].mkdir(exist_ok=True, parents=True)

        self.output_dpi = output_dpi
        self.output_transparency = output_transparency

    def __call__(self, name, fig=None):
        name = name.replace(" ", "_")
        if fig is None:
            fig = plt.gcf()
        fig.tight_layout()
        for ext, plot_dir in self.plot_dirs.items():
            fig.savefig(
                plot_dir / f"{name}.{ext}",
                bbox_inches="tight",
                dpi=self.output_dpi,
                transparent=self.output_transparency,
            )


evf = lambda S, f, **arg: (S, f(S, **arg))
