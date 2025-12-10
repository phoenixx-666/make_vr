from annotated_types import Ge, Gt, Le, Len
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, TypeAdapter, ValidationError
from typing import Annotated, Iterable, Iterator, TypeVar


__all__ = [
    'with_each',
    'duration',
    'NEStr',
    'NNFloat',
    'NNInt',
    'PosInt',
    'ImageQuality',
    'Angle',
    'SingletonMixin',
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
        return NNFloat(s)
    return NNFloat(sum(float(seg) * (60.0 ** i) for i, seg in enumerate(reversed(s.split(':')))))


class AnnotatedConverter:
    def __class_getitem__(cls, args: tuple):
        if isinstance(args[0], str):
            typename = args[0]
            args = args[1:]
        else:
            typename = args[0].__name__
        annotated_type = Annotated[*args]
        adapter = TypeAdapter(annotated_type)

        class _Converter:
            __annotated__ = annotated_type
            __origin__ = annotated_type
            __adapter = adapter
            __typename = typename

            def __call__(self, value):
                try:
                    return self.__adapter.validate_python(value)
                except ValidationError as e:
                    raise ValueError('\n'.join(err['msg'] for err in e.errors()))

            @staticmethod
            def __get_pydantic_core_schema__(_, handler: GetCoreSchemaHandler):
                return handler.generate_schema(annotated_type)

            def __repr__(self):
                return self.__typename

        return _Converter()


NEStr = AnnotatedConverter['non-empty string', str, Len(min_length=1)]
NNFloat = AnnotatedConverter['non-negative float', float, Ge(0.0)]
NNInt = AnnotatedConverter['non-negative int', int, Ge(0)]
PosInt = AnnotatedConverter['positive int', int, Gt(0)]
ImageQuality = AnnotatedConverter['image quality', int, Ge(1), Le(31)]
Angle = AnnotatedConverter['angle', float, Gt(0.0), Le(360.0)]


class SingletonMixin:
    __instance = None
    _need_init = True

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, *args, **kwargs):
        if self._need_init:
            super().__init__(*args, **kwargs)
            self._need_init = False


class ValidatedModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
