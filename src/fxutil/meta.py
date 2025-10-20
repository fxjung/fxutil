def in_ipython_session() -> bool:
    """
    Determine if we are running inside an IPython session
    (e.g., Jupyter Notebook or Jupyter Lab).
    """

    # stolen from https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/7
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
