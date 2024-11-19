import shlex

from .config import Config


__all__ = ['print_command']


def print_command(cfg: Config, command: list[str]):
    print(shlex.join(command))
    exit()
