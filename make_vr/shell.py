import shlex
import sys


__all__ = ['print_command', 'terminate']


def print_command(part1: list[str], fg: str, part2: list[str]):
    sys.stdout.write(shlex.join(part1))
    sys.stdout.write(f' {fg} ')
    sys.stdout.write(shlex.join(part2))
    sys.stdout.write('\n')
    exit()


def terminate(msg: str, exit_code=1):
    print(msg)
    exit(exit_code)