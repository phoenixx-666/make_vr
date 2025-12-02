from dataclasses import astuple, dataclass, field
import itertools
from io import StringIO
import shlex
import sys
from typing import ClassVar, TextIO

from .filters import FilterGraph

__all__ = ['FFMpegCommand', 'print_command', 'terminate']


@dataclass
class FFMpegCommand:
    @dataclass
    class Output:
        mappings: list[str] = field(default_factory=list)
        codecs: list[str] = field(default_factory=list)
        outputs: list[str] = field(default_factory=list)

    ffmpeg_executable: str
    general_params: list[str] = field(default_factory=list)
    inputs: list[list[str]] = field(default_factory=list)
    filter_complex: ClassVar[str] = '-filter_complex'
    filter_graph: FilterGraph | None = None
    outputs: list[Output] = field(default_factory=list)

    def as_list(self) -> list[str]:
        params = [self.ffmpeg_executable]
        params.extend(self.general_params)
        for input in self.inputs:
            params.extend(input)
        if self.filter_graph is not None:
            params.extend([self.filter_complex])
            params.extend([self.filter_graph.render()])
        for output in self.outputs:
            params.extend(itertools.chain.from_iterable(astuple(output)))
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
            buffer.write(f" '{self.filter_graph.render(True, indent=INDENT * 2)}'")
        buffer.write(HYPHEN)
        buffer.write(ENDL)
        for i, output in enumerate(self.outputs):
            for j, section in enumerate(sections_tuple := astuple(output)):
                buffer.write(INDENT)
                buffer.write(shlex.join(section))
                if (i != (len(self.outputs) - 1)) or (j != (len(sections_tuple) - 1)):
                    buffer.write(HYPHEN)
                buffer.write(ENDL)

        stream.write(buffer.getvalue())


def print_command(part1: list[str], fg: str, part2: list[str]):
    sys.stdout.write(shlex.join(part1))
    sys.stdout.write(f' {fg} ')
    sys.stdout.write(shlex.join(part2))
    sys.stdout.write('\n')
    exit()


def terminate(msg: str, exit_code=1):
    sys.stderr.write(f'\x1b[31m{msg}\x1b[0m\n')
    exit(exit_code)
