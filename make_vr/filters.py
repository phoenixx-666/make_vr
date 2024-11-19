from dataclasses import dataclass, field
from itertools import chain
import re
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
        return cls._escape_re.sub(r'\\$0', s)

    @classmethod
    def _render_param(cls, param: Any) -> str:
        if isinstance(param, int):
            return f'{param:d}'
        elif isinstance(param, float):
            return fts(param)
        elif isinstance(param, str):
            return cls._escape_string(param)
        return cls._escape_string(str(param))

    def render(self) -> str:
        if not self.simple_params and not self.kw_params:
            return self._name
        simple_params = map(self._render_param, self.simple_params)
        kw_params = (f'{k}={self._render_param(v)}' for k, v in self.kw_params.items())
        return f'{self._name}={":".join(chain.from_iterable([simple_params, kw_params]))}'


@dataclass
class FilterSeq:
    inputs: list[str]
    outputs: list[str]
    filters: list[Filter] = field(default_factory=list)

    def render(self) -> str:
        inputs = (f'[{input}]' for input in self.inputs)
        outputs = (f'[{output}]' for output in self.outputs)
        return f'{"".join(inputs)}{",".join(map(Filter.render, self.filters))}{"".join(outputs)}'


@dataclass
class FilterGraph:
    filter_seqs: list[FilterSeq]

    def render(self) -> str:
        return ','.join(map(FilterSeq.render, self.filter_seqs))
