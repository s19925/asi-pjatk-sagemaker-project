"""asi-pjatk-sagemaker-project file for ensuring the package is executable
as `asi-pjatk-sagemaker-project` and `python -m asi_pjatk_sagemaker_project`
"""
import importlib
from pathlib import Path
import pandas as pd
import wandb
import pipelines.data_processing.nodes as pdp
import pipelines.data_science.nodes as pds
from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project


def _find_run_command(package_name):
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
        # fail gracefully if cli.py does not exist
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = _find_run_command_in_plugins(plugins) if plugins else None
        if run:
            # use run command from installed plugin if it exists
            return run
        # use run command from the framework project
        from kedro.framework.cli.project import run

        return run
    # fail badly if cli.py exists, but has no `cli` in it
    if not hasattr(project_cli, "cli"):
        raise KedroCliError(f"Cannot load commands from {package_name}.cli")
    return project_cli.run


def _find_run_command_in_plugins(plugins):
    for group in plugins:
        if "run" in group.commands:
            return group.commands["run"]



def main(*args, **kwargs):
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = _find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main()

