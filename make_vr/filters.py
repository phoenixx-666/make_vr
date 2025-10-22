from dataclasses import dataclass, field
from itertools import chain
import re
import shlex
from typing import Any


__all__ = ['Filter', 'FilterSeq', 'FilterGraph', 'fts']


def fts(f: float) -> str:
    return f'{f:.9f}'.rstrip('0').rstrip('.')


class Filter:
    _escape_re = re.compile(r'''[:,='"\\]''')

    def __init__(self, name=str, *simple_params, **kw_params):
        self._name = name
        self.simple_params = simple_params
        self.kw_params = kw_params
        self.raw_params = []

    """
    @staticmethod
    def _escape_str(s: str) -> str:
        sio = io.StringIO()
        for c in s:
            if c in r',:\\':
                sio.write('\\')
            sio.write(c)
        return sio.getvalue()
    """

    @classmethod
    def _escape_string(cls, s: str) -> str:
        return cls._escape_re.sub(r'\\\g<0>', s)

    @classmethod
    def _render_param(cls, param: Any, escape=True) -> str:
        if isinstance(param, int):
            return f'{param:d}'
        elif isinstance(param, float):
            return fts(param)

        if not isinstance(param, str):
            param = str(param)
        if escape:
            return cls._escape_string(param)
        else:
            return param

    def add_raw(self, raw_params: str):
        self.raw_params.append(raw_params)

    def render(self) -> str:
        if not self.simple_params and not self.kw_params:
            return self._name
        simple_params = map(self._render_param, self.simple_params)
        kw_params = (f'{k}={self._render_param(v)}' for k, v in self.kw_params.items())
        raw_params = [self._render_param(param, escape=False) for param in self.raw_params]
        return f'{self._name}={":".join(chain.from_iterable([simple_params, kw_params, raw_params]))}'


@dataclass
class FilterSeq:
    inputs: list[str]
    outputs: list[str]
    filters: list[Filter] = field(default_factory=list)

    def render(self) -> str:
        inputs = (f'[{input}]' for input in self.inputs)
        outputs = (f'[{output}]' for output in self.outputs)
        rendered_filters = map(Filter.render, self.filters)
        return f'{"".join(inputs)}{",".join(rendered_filters)}{"".join(outputs)}'


@dataclass
class FilterGraph:
    filter_seqs: list[FilterSeq] = field(default_factory=list)

    def render(self, for_print = False, indent: str = '') -> str:
        if for_print:
            return self._render_for_print(indent)
        else:
            return self._render_for_exec()

    def _render_for_print(self, indent: str) -> str:
        def format(filter_seq: str) -> str:
            return f'{indent}{filter_seq}'

        rendered_seqs = map(FilterSeq.render, self.filter_seqs)
        formatted_seqs = map(format, rendered_seqs)
        return '\n' + ',\n'.join(formatted_seqs) + '\n'

    def _render_for_exec(self) -> str:
        return ','.join(map(FilterSeq.render, self.filter_seqs))
