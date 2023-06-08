from typer.testing import CliRunner
from pyinterpolate import __app_name__, __version__, cli


RUNNER = CliRunner()


def test_version():
    result = RUNNER.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in result.stdout
