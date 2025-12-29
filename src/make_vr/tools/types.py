from annotated_types import Ge, Gt, Le, Len
from typing import Annotated

from pydantic import GetCoreSchemaHandler, TypeAdapter, ValidationError


__all__ = [
    'AnnotatedConverter',
    'NEStr',
    'NNFloat',
    'NNInt',
    'PosInt',
    'ImageQuality',
    'Angle',
]


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
