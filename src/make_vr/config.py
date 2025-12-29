from pydantic import Field, ValidationError
from typing import Any, ClassVar, Optional

from .shell import terminate
from .tools import SingletonModel, ValidatedModel
from .tools.types import Angle, ImageQuality, NEStr, NNInt, PosInt


__all__ = ['Config']


class Config(ValidatedModel, metaclass=SingletonModel):
    class Options(ValidatedModel):
        threads: Optional[NNInt] = None

    class Inputs(ValidatedModel):
        left_dir: Optional[NEStr] = Field(alias='left-dir', default=None)
        right_dir: Optional[NEStr] = Field(alias='right-dir', default=None)

    class Encoding(ValidatedModel):
        bitrate: NEStr = '40M'
        video_codec: NEStr = Field(alias='video-codec', default='hevc_nvenc')
        preset: NEStr = 'slow'
        pixel_format: NEStr = Field(alias='pixel-format', default='yuv420p')
        x264_frame_packing: bool = Field(alias='x264-frame-packing', default=False)
        audio_codec: NEStr = Field(alias='audio-codec', default='aac')
        audio_bitrate: NEStr = Field(alias='audio-bitrate', default='256k')
        image_quality: ImageQuality = Field(alias='image-quality', default=2)

    class Camera(ValidatedModel):
        ihfov: Angle = 122.0
        ivfov: Angle = 108.0

    class Processing(ValidatedModel):
        video_filter: list[NEStr] = Field(alias='video-filter', default_factory=list)
        audio_filter: list[NEStr] = Field(alias='audio-filter', default_factory=list)

    class Visuals(ValidatedModel):
        ohfov: Angle = 180.0
        ovfov: Angle = 180.0
        ow: Optional[PosInt] = None
        oh: Optional[PosInt] = None
        of: NEStr = 'e'

    options: Options = Field(default_factory=Options)
    inputs: Inputs = Field(default_factory=Inputs)
    encoding: Encoding = Field(default_factory=Encoding)
    camera: Camera = Field(default_factory=Camera)
    processing: Processing = Field(default_factory=Processing)
    visuals: Visuals = Field(default_factory=Visuals)

    _error_prefix: ClassVar[str] = 'Error while validating config: '

    def load(self, data: dict[str, Any]):
        def process_dict(data: dict[str, Any], obj: ValidatedModel):
            for k, v in data.items():
                k = k.replace('-', '_')
                if isinstance(v, dict):
                    process_dict(v, getattr(obj, k))
                else:
                    setattr(obj, k, v)

        if not isinstance(data, dict):
            terminate(f'{self._error_prefix}wrong data structure')

        if not (defaults := data.get('defaults')):
            return

        try:
            process_dict(defaults, self)
        except AttributeError as e:
            terminate(f'{self._error_prefix}{e.args[0]}')
        except ValidationError as e:
            terminate(f'{self._error_prefix}{"\n".join(err["msg"] for err in e.errors())}')
