from typing import Optional
import typer

from pyinterpolate import __app_name__, __version__
from pyinterpolate.cli_utils.controllers.variogram import ExperimentalVariogramController

app = typer.Typer()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show the application's version and exit.",
            callback=_version_callback,
            is_eager=True,
        )
) -> None:
    return


@app.command()
def get_semivariogram(
        fpath: str,
        step_size: float,
        max_range: float,
        direction: float = None,
        tolerance: float = 1,
        method='t'
) -> None:
    """
    Calculate experimental semivariogram from a given set of points.

    Returns
    -------

    """
    _ = ExperimentalVariogramController(
        input_array_file=fpath,
        step_size=step_size,
        max_range=max_range,
        direction=direction,
        tolerance=tolerance,
        method=method
    )
