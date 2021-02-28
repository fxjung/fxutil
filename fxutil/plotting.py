import matplotlib.pyplot as plt
from pathlib import Path


class SaveFigure:
    def __init__(self, plot_dir):
        plot_dir = Path(plot_dir)
        self.plot_dirs = {}
        for ext in ["pdf", "png"]:
            self.plot_dirs[ext] = plot_dir / ext
            self.plot_dirs[ext].mkdir(exist_ok=True, parents=True)

    def __call__(self, name, fig=None):
        name = name.replace(" ", "_")
        if fig is None:
            fig = plt.gcf()
        fig.tight_layout()
        for ext, plot_dir in self.plot_dirs.items():
            fig.savefig(plot_dir / f"{name}.{ext}", bbox_inches="tight", dpi=160)
