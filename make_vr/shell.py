import shlex
import sys

from .config import Config


__all__ = ['print_command']


def print_command(cfg: Config, part1: list[str], fg: str, part2: list[str]):
    sys.stdout.write(shlex.join(part1))
    sys.stdout.write(f' {fg} ')
    sys.stdout.write(shlex.join(part2))
    sys.stdout.write('\n')
    exit()
