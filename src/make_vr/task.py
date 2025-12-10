from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any, Self

from .config import Config
from .fs import validate_input_files
from .shell import CLIArgs, terminate


__all__ = ['Task']


@dataclass
class Task:
    threads: int | None
    overwrite: bool
    rename: bool
    ask_match: bool
    print: int | None
    do_print: bool

    bitrate: str
    video_codec: str
    preset: str
    pixel_format: str
    x264_frame_packing: bool
    audio_codec: str
    audio_bitrate: str

    quality: int
    do_image: bool

    ffmpeg_path: str
    ffprobe_path: str
    ffmpeg_verbose: bool

    @dataclass
    class Segment:
        left: str | list[str]
        right: str | list[str]
        audio: str | list[str]

        external_audio: bool

        duration: float | None

        ih_fov: float
        iv_fov: float

        fade: list[float]
        channel: int
        do_audio: bool

        extra_offset: float | None
        override_offset: float | None
        trim: float | None
        ultra_sync: int | None
        extra_video_filter: list[str]
        extra_audio_filter: list[str]
        fill_end: bool
        wav_duration: float | None

    segments: list[Segment]
    output: str | None
    separate_audio: bool
    do_stab: bool
    stab_args: str | None
    stab_channel: int

    oh_fov: float
    ov_fov: float
    ow: int | None
    oh: int | None
    of: str

    @classmethod
    def from_args(cls) -> Self:

        if (args := CLIArgs()).config:
            config_name = args.config
            if not os.path.isfile(config_name):
                terminate('Specified config path is not a file')
            load_config = True
        else:
            config_name = str(Path(os.getcwd()) / 'make_vr.toml')
            load_config = os.path.isfile(config_name)

        if load_config:
            try:
                with open(config_name, 'rb') as f:
                    config_data = tomllib.load(f)
            except (IOError, OSError, tomllib.TOMLDecodeError) as e:
                terminate(f'Error while reading config: {e}')
            Config().load(config_data)

        arg_mult = _ArgsMultiplier(len((inputs := validate_input_files()).segments))

        segments = [
            cls.Segment(
                left=inputs.segments[i].left,
                right=inputs.segments[i].right,
                audio=inputs.segments[i].audio,

                external_audio=bool(inputs.segments[i].audio),

                duration=arg_mult.multiply_arg('duration', None, i),

                ih_fov=arg_mult.multiply_arg('ihfov', Config().camera.ihfov, i),
                iv_fov=arg_mult.multiply_arg('ivfov', Config().camera.ivfov, i),

                fade=arg_mult.multiply_arg('fade', [1.0], i),
                channel=(channel := arg_mult.multiply_arg('channel', 0, i)),
                do_audio=channel != -1,

                extra_offset=arg_mult.multiply_arg('offset', 0.0, i),
                override_offset=arg_mult.multiply_arg('override-offset', None, i),
                trim=arg_mult.multiply_arg('trim', 0.0, i),
                ultra_sync=arg_mult.multiply_arg('ultra-sync', None, i),
                extra_video_filter=arg_mult.multiply_arg('video-filter', Config().processing.video_filter, i),
                extra_audio_filter=arg_mult.multiply_arg('audio-filter', Config().processing.audio_filter, i),
                fill_end=arg_mult.multiply_arg('fill-end', False, i),
                wav_duration=arg_mult.multiply_arg('wav-duration', None, i),

            )
            for i in range(arg_mult._num_segments)
        ]

        return Task(
            threads=args.threads if args.threads is not None else Config().options.threads,
            overwrite=bool(args.overwrite or args.yes),
            rename=bool(args.rename),
            ask_match=bool(args.ask_match),
            print=args.print,
            do_print=args.print is not None,

            bitrate=args.bitrate or Config().encoding.bitrate,
            video_codec=args.video_codec or Config().encoding.video_codec,
            preset=args.preset or Config().encoding.preset,
            pixel_format=args.pixel_format or Config().encoding.pixel_format,
            x264_frame_packing=args.x264_frame_packing or Config().encoding.x264_frame_packing,
            audio_codec=args.audio_codec or Config().encoding.audio_codec,
            audio_bitrate=args.audio_bitrate or Config().encoding.audio_bitrate,

            segments=segments,
            output=args.output,

            separate_audio=args.separate_audio,

            do_stab=inputs.do_stab,
            stab_args=args.stab if args.stab is not ... else None,
            stab_channel=0 if (args.stab_channel is None) else args.stab_channel,

            quality=args.quality or Config().encoding.image_quality,
            do_image=inputs.do_image,

            oh_fov=args.ohfov or Config().visuals.ohfov,
            ov_fov=args.ovfov or Config().visuals.ovfov,
            ow=args.ow or Config().visuals.ow,
            oh=args.oh or Config().visuals.oh,
            of=args.of or Config().visuals.of,

            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
            ffmpeg_verbose=args.ffmpeg_print,
        )


class _ArgsMultiplier:
    def __init__(self, num_segments: int):
        self._args = CLIArgs()
        self._num_segments = num_segments

    def multiply_arg(self, arg_name: str, default: Any, index: int) -> Any:
        if (arg := getattr(self._args, arg_name.replace('-', '_'))) is None:
            return default

        if len(arg) == self._num_segments:
            return arg[index]

        if len(arg) == 1:
            return arg[0]

        terminate(f'Argument "{arg_name}" must be specified either for every segment or once')
