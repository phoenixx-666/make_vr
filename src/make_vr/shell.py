import argparse
from collections import defaultdict
from dataclasses import astuple, dataclass, field
from enum import Enum, auto
import itertools
from io import StringIO
import shlex
import sys
from typing import ClassVar, TextIO

from .filters import FilterGraph
from .tools import Singleton, duration
from .tools.types import Angle, ImageQuality, NEStr, NNInt, PosInt


__all__ = ['CLIArgs', 'FFMpegCommand', 'LogLevel', 'message', 'info', 'success', 'warning', 'error', 'terminate']


_DEFAULT_USYNC = 5


class CLIArgs(argparse.Namespace, metaclass=Singleton):
    def __init__(self):
        args = self._make_parser().parse_args()
        super().__init__(**dict(args._get_kwargs()))

    @staticmethod
    def _make_parser() -> argparse.ArgumentParser:
        from .config import Config
        parser = argparse.ArgumentParser()

        parser.description = 'VR Video maker designed for processing GoPro fisheye videos'

        parser.add_argument('-C', '--config', type=NEStr, metavar='path')

        parser.add_argument('-l', '--left', action='append', type=NEStr, nargs='+', metavar='path')
        parser.add_argument('-r', '--right', action='append', type=NEStr, nargs='+', metavar='path')
        parser.add_argument('-a', '--audio', action='append', type=NEStr, nargs='*', metavar='path', help='External audio file')
        parser.add_argument('-o', '--output', type=NEStr, metavar='path')

        parser.add_argument('-t', '--threads', type=NNInt)
        parser.add_argument('--ignore-cache', type=str, choices=['match', 'sync', 'save'], nargs='*',
                            help='Do not perform operation with cached data. No args means ignoring cache completely')
        parser.add_argument('-y', dest='yes', action='store_true', help='Answer "Yes" to all prompts involving output files')
        mex_group = parser.add_mutually_exclusive_group()
        mex_group.add_argument('-w', '--overwrite', action='store_true', help='Overwrite without prompt if output file already exists')
        mex_group.add_argument('--rename', action='store_true', help='Overwrite without prompt if output file already exists')
        parser.add_argument('-m', '--ask-match', action='store_true', help='Prompt when automatically finding matching file by date')
        parser.add_argument('-p', '--print', type=NNInt, nargs='?', const=0, metavar='line_width', help='Only print ffmpeg command without executing it')

        arg_group = parser.add_argument_group('Camera options')
        arg_group.add_argument('--ihfov', action='append', type=Angle, help='Horizontal FOV of an input fisheye visual')
        arg_group.add_argument('--ivfov', action='append', type=Angle, help='Vertical FOV of an input fisheye visual')

        arg_group = parser.add_argument_group('Video processing options')
        arg_group.add_argument('-f', '--fade', action='append', type=duration, nargs='+', help='Add fade effect in the beginning and the end of the video segment')
        arg_group.add_argument('-c', '--channel', action='append', type=int, choices=[-1, 0, 1],
                               help='Use audio from the video file segment for the given eye, or -1 for no audio')
        arg_group.add_argument('-d', '--duration', action='append', type=duration, help='Maximum duration of the output video segment [seconds or h:m:s]')
        arg_group.add_argument('--offset', action='append', type=duration, help='Extra offset for the cut video segment [seconds or h:m:s]')
        arg_group.add_argument('--override-offset', action='append', nargs=2, metavar=('INPUT', 'OFFSET'), help='Override offset on specified input of video segment')
        arg_group.add_argument('--trim', '-T', action='append', type=duration, help='Time to trim video segment from the beginning, after the videos are synchronized')
        arg_group.add_argument('--ultra-sync', '-u', action='append', type=NNInt, nargs='?', const=_DEFAULT_USYNC,
                                help=f'Apply extra synchronization by multiplying FPS by given number [default: {_DEFAULT_USYNC}], '
                                     'with interpolation and then lowering FPS back after trimming. '
                                     'For multiple segment videos, specify 0 as the argument to suppress sync on current segment')
        arg_group.add_argument('--video-filter', '-F', action='append', type=NEStr, nargs='*', metavar='filter',
                                help='Custom video filters, applied before conversion from fisheye to equirectangular')
        arg_group.add_argument('--audio-filter', '--af', action='append', type=NEStr, nargs='*', metavar='filter', help='Custom audio filters')
        arg_group.add_argument('--fill-end', action='append', type=int, choices=[0, 1], nargs='?', const=1,
                               help='If two video segments are of unequal length after synchronization, '
                                    'fill the end of the resulting segment with non-stereo content, '
                                    'up to the duration of the longer video. For multiple-segment videos, '
                                    'specify "--fill-end 0" for the segments that require no filling')
        arg_group.add_argument('--wav-duration', type=duration, metavar='DURATION',
                               help='Portion of sound to extract from left and right video that will be used in automatic synchronization')

        arg_group = parser.add_argument_group('Video stabilization options')
        arg_group.add_argument('--stab', type=NEStr, metavar='ARGS', nargs='?', const=...,
                               help='Produce the stabilization file instead of video with vidstabdetect filter with optional extra ARGS')
        arg_group.add_argument('--stab-channel', type=int, choices=[0, 1], metavar='INDEX',
                               help='Index of the input channel to produce stabilization file from')

        arg_group = parser.add_argument_group('Video encoding options')
        arg_group.add_argument('--video-codec', '--vc', type=NEStr,
                               help=f'Video codec (must be suitable for FFMpeg, default: {Config().encoding.video_codec})')
        arg_group.add_argument('-b', '--bitrate', type=NEStr, help=f'Output video bitrate (Value must be compatible with FFMpeg, default: {Config().encoding.bitrate})')
        arg_group.add_argument('--preset', type=NEStr, help=f'Encoding preset (must be suitable for FFMpeg, default: {Config().encoding.preset})')
        arg_group.add_argument('--pixel-format', '--pf', type=NEStr,
                               help=f'Pixel format codec (must be suitable for FFMpeg, default: {Config().encoding.pixel_format})')
        arg_group.add_argument('--x264-frame-packing', '--x264fp', action='store_true', help='Apply x264 frame packing')
        arg_group.add_argument('--audio-codec', '--ac', type=NEStr,
                               help=f'Audio codec (must be suitable for FFMpeg, default: {Config().encoding.audio_codec})')
        arg_group.add_argument('--audio-bitrate', '--ab', type=NEStr,
                               help=f'Audio bitrate (must be suitable for FFMpeg, default: {Config().encoding.audio_bitrate})')
        arg_group.add_argument('--separate-audio', '-A', action='store_true', help='Store audio as a separate wav file for further editing')

        arg_group = parser.add_argument_group('Image encoding options')
        arg_group.add_argument('-q', '--quality', type=ImageQuality, help=f'Output image quality, default: {Config().encoding.image_quality}')

        arg_group = parser.add_argument_group('Visual format options')
        arg_group.add_argument('--ohfov', type=Angle, help='Horizontal FOV of the output visual')
        arg_group.add_argument('--ovfov', type=Angle, help='Vertical FOV of the output visual')
        arg_group.add_argument('--ow', type=PosInt, help='Horizontal resolution of the single-eye part of the visual')
        arg_group.add_argument('--oh', type=PosInt, help='Vertical resolution of the single-eye part of the visual')
        arg_group.add_argument('--of', type=NEStr, help='Output format of the visual (equirectangular, flat, etc...)')

        arg_group = parser.add_argument_group('Other options')
        arg_group.add_argument('--ffmpeg-path', type=NEStr, default='ffmpeg', help='Path to the ffmpeg executable')
        arg_group.add_argument('--ffprobe-path', type=NEStr, default='ffprobe', help='Path to the ffprobe executable')
        arg_group.add_argument('--ffmpeg-print', '-P', action='store_true', help='Print all ffmpeg output')

        return parser


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


class LogLevel(Enum):
    Info = auto()
    Success = auto()
    Warning = auto()
    Error = auto()


_log_level_to_color = defaultdict(int, {
    LogLevel.Info: 93,
    LogLevel.Success: 92,
    LogLevel.Warning: 33,
    LogLevel.Error: 31,
})


def message(msg: str, end: str = ..., level = LogLevel.Info):
    if end is ...:
        end = '\n'
    elif end is None:
        end = ''
    color = _log_level_to_color[level]
    sys.stderr.write(f'\033[{color}m{msg}\033[0m{end}')
    sys.stderr.flush()


def info(msg: str, end: str = ...):
    message(msg, end, LogLevel.Info)


def success(msg: str, end: str = ...):
    message(msg, end, LogLevel.Success)


def warning(msg: str, end: str = ...):
    message(msg, end, LogLevel.Warning)


def error(msg: str, end: str = ...):
    message(msg, end, LogLevel.Error)


def terminate(msg: str, exit_code=1, end: str = ..., level = LogLevel.Error):
    message(msg, end, level)
    exit(exit_code)
