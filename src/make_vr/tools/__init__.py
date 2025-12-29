from pydantic import BaseModel, ConfigDict
from pydantic._internal._model_construction import ModelMetaclass
from typing import Iterable, Iterator, TypeVar


from .types import NNFloat as _NNFLoat


__all__ = [
    'with_each',
    'duration',
    'Singleton',
    'SingletonModel',
    'ValidatedModel',
]


_T1 = TypeVar('T1')
_T2 = TypeVar('T2')


class with_each(Iterable[tuple[_T1, _T2]]):
    class Iterator(Iterator[tuple[_T1, _T2]]):
        def __init__(self, item: _T1, iterable: Iterable[_T2]):
            self._gen = ((item, elem) for elem in iterable)

        def __next__(self) -> tuple[_T1, _T2]:
            return next(self._gen)

    def __init__(self, item: _T1, iterable: Iterable[_T2]):
        self._item = item
        self._iterable = iterable

    def __iter__(self):
        return self.Iterator(self._item, self._iterable)


def duration(s: str | float | int) -> float:
    if isinstance(s, (float, int)):
        return _NNFLoat(s)
    return _NNFLoat(sum(float(seg) * (60.0 ** i) for i, seg in enumerate(reversed(s.split(':')))))


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonModel(ModelMetaclass, Singleton):
    pass


class ValidatedModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
