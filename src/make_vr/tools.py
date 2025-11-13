from typing import Iterable, Iterator, TypeVar


__all__ = ['with_each']


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
