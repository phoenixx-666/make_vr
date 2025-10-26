from dataclasses import dataclass, field
from io import StringIO
import shlex
import sys
from typing import TextIO

from .filters import FilterGraph

__all__ = ['FFMpegCommand', 'print_command', 'terminate']


@dataclass
class FFMpegCommand:
    ffmpeg_executable: str
    general_params: list[str] = field(default_factory=list)
    inputs: list[list[str]] = field(default_factory=list)
    filter_complex: str = '-filter_complex'
    filter_graph: FilterGraph | None = None
    codecs_and_outputs: list[str] = field(default_factory=list)

    def as_list(self) -> list[str]:
        params = [self.ffmpeg_executable]
        params.extend(self.general_params)
        for input in self.inputs:
            params.extend(input)
        if self.filter_graph is not None:
            params.extend([self.filter_complex])
            params.extend([self.filter_graph.render()])
        params.extend(self.codecs_and_outputs)
        return params

    def print(self, stream: TextIO = sys.stdout, indent: int = 2):
        INDENT = ' ' * indent
        HYPHEN = ' \\'
        ENDL = '\n'

        buffer = StringIO()
        buffer.write(shlex.join([self.ffmpeg_executable] + self.general_params))
        buffer.write(HYPHEN)
        buffer.write(ENDL)
        for input in self.inputs:
            buffer.write(INDENT * 2)
            buffer.write(shlex.join(input))
            buffer.write(HYPHEN)
            buffer.write(ENDL)
        if self.filter_graph is not None:
            buffer.write(INDENT)
            buffer.write(self.filter_complex)
            buffer.write(f" '{self.filter_graph.render(True, indent=INDENT * 2)}' ")
        buffer.write(shlex.join(self.codecs_and_outputs))
        buffer.write(ENDL)

        stream.write(buffer.getvalue())


def print_command(part1: list[str], fg: str, part2: list[str]):
    sys.stdout.write(shlex.join(part1))
    sys.stdout.write(f' {fg} ')
    sys.stdout.write(shlex.join(part2))
    sys.stdout.write('\n')
    exit()


def terminate(msg: str, exit_code=1):
    print(msg)
    exit(exit_code)
