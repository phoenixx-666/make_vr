import argparse
from dataclasses import dataclass
from typing import Any, Self

from .fs import validate_input_files
from .shell import terminate


__all__ = ['Config', 'duration']


_DEFAULT_USYNC = 5


def duration(s: str | float | int) -> float:
    if isinstance(s, (float, int)):
        return float(s)
    return sum(float(seg) * (60.0 ** i) for i, seg in enumerate(reversed(s.split(':'))))


@dataclass
class Config:
    threads: int
    get_offset: bool
    overwrite: bool
    rename: bool
    ask_match: bool
    print: int | None
    do_print: bool

    bitrate: str
    video_codec: str
    preset: str
    pixel_format: str
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

        stab_args: str | None
        stab_channel: int = 0

    segments: list[Segment]
    output: str | None
    separate_audio: bool
    do_stab: bool

    @classmethod
    def from_args(cls) -> Self:
        args = cls._make_parser().parse_args()

        do_stab = (args.stab is not None) or (args.stab_channel is not None)
        do_image, left, right, audio = validate_input_files(args.ffprobe_path,
                                                            args.left, args.right, args.audio or [],
                                                            args.ask_match, do_stab)

        am = _ArgsMultiplier(len(left), args)

        segments = [
            cls.Segment(
                left=left[i],
                right=right[i],
                audio=audio[i],

                external_audio=bool(audio[i]),

                duration=am.multiply_arg('duration', None, i),

                ih_fov=am.multiply_arg('ihfov', 122.0, i),
                iv_fov=am.multiply_arg('ivfov', 108.0, i),

                fade=am.multiply_arg('fade', [1.0], i),
                channel=(channel := am.multiply_arg('channel', 0, i)),
                do_audio=channel != -1,

                extra_offset=am.multiply_arg('offset', 0.0, i),
                override_offset=am.multiply_arg('override-offset', None, i),
                trim=am.multiply_arg('trim', 0.0, i),
                ultra_sync=am.multiply_arg('ultra-sync', None, i),
                extra_video_filter=am.multiply_arg('video-filter', [], i),
                extra_audio_filter=am.multiply_arg('audio-filter', [], i),
                fill_end=am.multiply_arg('fill-end', False, i),
                wav_duration=am.multiply_arg('wav-duration', None, i),

                stab_args=args.stab or None,
                stab_channel=0 if (args.stab_channel is None) else args.stab_channel,
            )
            for i in range(am.num_segments)
        ]

        return Config(
            threads=args.threads,
            get_offset=bool(args.get_offset),
            overwrite=bool(args.overwrite or args.yes),
            rename=bool(args.rename),
            ask_match=bool(args.ask_match),
            print=args.print,
            do_print=args.print is not None,

            bitrate=args.bitrate,
            video_codec=args.video_codec,
            preset=args.preset,
            pixel_format=args.pixel_format,
            audio_codec=args.audio_codec,
            audio_bitrate=args.audio_bitrate,

            segments=segments,
            output=args.output,

            separate_audio=args.separate_audio,

            do_stab=do_stab,

            quality=args.quality,
            do_image=do_image,

            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
            ffmpeg_verbose=args.ffmpeg_print,
        )

    @staticmethod
    def _make_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()

        parser.description = 'VR Video maker designed for processing GoPro fisheye videos'
        parser.add_argument('-l', '--left', action='append', type=str, nargs='+', metavar='path', required=True)
        parser.add_argument('-r', '--right', action='append', type=str, nargs='+', metavar='path', required=True)
        parser.add_argument('-a', '--audio', action='append', type=str, nargs='*', metavar='path', help='')
        parser.add_argument('-o', '--output', type=str, metavar='path')

        parser.add_argument('-t', '--threads', type=int)
        parser.add_argument('--get-offset', action='store_true', help='Only calculate offset between files based on the audio tracks')
        parser.add_argument('-y', dest='yes', action='store_true', help='Answer "Yes" to all prompts involving output files')
        mex_group = parser.add_mutually_exclusive_group()
        mex_group.add_argument('-w', '--overwrite', action='store_true', help='Overwrite without prompt if output file already exists')
        mex_group.add_argument('--rename', action='store_true', help='Overwrite without prompt if output file already exists')
        parser.add_argument('-m', '--ask-match', action='store_true', help='Prompt when automatically finding matching file by date')
        parser.add_argument('-p', '--print', type=int, nargs='?', const=0, metavar='line_width', help='Only print ffmpeg command without executing it')

        arg_group = parser.add_argument_group('Camera options')
        arg_group.add_argument('--ihfov', action='append', type=float, help='Horizontal FOV of an input fisheye image')
        arg_group.add_argument('--ivfov', action='append', type=float, help='Vertical FOV of an input fisheye image')

        arg_group = parser.add_argument_group('Video processing options')
        arg_group.add_argument('-f', '--fade', action='append', type=duration, nargs='+', help='Add fade effect in the beginning and the end of the video segment')
        arg_group.add_argument('-c', '--channel', action='append', type=int, choices=[-1, 0, 1], help='Use audio from the video file segment for the given eye, or -1 for no audio')
        arg_group.add_argument('-d', '--duration', action='append', type=duration, help='Maximum duration of the output video segment [seconds or h:m:s]')
        arg_group.add_argument('--offset', action='append', type=duration, help='Extra offset for the cut video [seconds or h:m:s]')
        arg_group.add_argument('--override-offset', action='append', nargs=2, metavar=('INPUT', 'OFFSET'), help='Override offset on specified input of video segment')
        arg_group.add_argument('--trim', '-T', action='append', type=duration, help='Time to trim video segment from the beginning, after the videos are synchronized')
        arg_group.add_argument('--ultra-sync', '-u', action='append', type=int, nargs='?', const=_DEFAULT_USYNC,
                                help=f'Apply extra synchronization by multiplying FPS by given number [default: {_DEFAULT_USYNC}], '
                                     'with interpolation and then lowering FPS back after trimming. '
                                     'For multiple segment videos, specify 0 as the argument to suppress sync on current segment')
        arg_group.add_argument('--video-filter', '-F', action='append', type=str, nargs='*', metavar='filter',
                                help='Custom video filters, applied before conversion from fisheye to equirectangular')
        arg_group.add_argument('--audio-filter', '--af', action='append', type=str, nargs='*', metavar='filter', help='Custom audio filters')
        arg_group.add_argument('--fill-end', action='append', type=int, nargs='?', const=1,
                               help='If two video segments are of unequal length after synchronization, '
                                    'fill the end of the resulting segment with non-stereo content, '
                                    'up to the duration of the longer video. For multiple-segment videos, '
                                    'specify "--fill-end 0" for the segments that require no filling')
        arg_group.add_argument('--wav-duration', type=duration, metavar='DURATION',
                               help='Portion of sound to extract from left and right video that will be used in automatic synchronization')

        arg_group = parser.add_argument_group('Video options')
        arg_group.add_argument('--video-codec', '--vc', type=str, default=(DEFAULT := 'hevc_nvenc'),
                               help=f'Video codec (must be suitable for FFMpeg, default: {DEFAULT})')
        arg_group.add_argument('-b', '--bitrate', type=str, default='40M', help='Output video bitrate (Value must be compatible with ffmpeg)')
        arg_group.add_argument('--preset', type=str, default=(DEFAULT := 'slow'), help=f'Encoding preset (must be suitable for FFMpeg, default: {DEFAULT})')
        arg_group.add_argument('--pixel-format', '--pf', type=str, default=(DEFAULT := 'yuv420p'),
                               help=f'Pixel format codec (must be suitable for FFMpeg, default: {DEFAULT})')
        arg_group.add_argument('--audio-codec', '--ac', type=str, default=(DEFAULT := 'aac'),
                               help=f'Audio codec (must be suitable for FFMpeg, default: {DEFAULT})')
        arg_group.add_argument('--audio-bitrate', '--ab', type=str, default=(DEFAULT := '192k'),
                               help=f'Audio bitrate (must be suitable for FFMpeg, default: {DEFAULT})')
        arg_group.add_argument('--separate-audio', '-A', action='store_true', help='Store audio as a separate wav file for further editing')
        arg_group.add_argument('--stab', type=str, metavar='ARGS', nargs='?', const='',
                               help='Produce the stabilization file instead of video with vidstabdetect filter with optional extra ARGS')
        arg_group.add_argument('--stab-channel', type=int, metavar='INDEX',
                               help='Index of the input channel to produce stabilization file from')

        arg_group = parser.add_argument_group('Photo options')
        arg_group.add_argument('-q', '--quality', type=int, default=2, help='Output photo quality')

        arg_group = parser.add_argument_group('Other options')
        arg_group.add_argument('--ffmpeg-path', type=str, default='ffmpeg', help='Path to the ffmpeg executable')
        arg_group.add_argument('--ffprobe-path', type=str, default='ffprobe', help='Path to the ffprobe executable')
        arg_group.add_argument('--ffmpeg-print', '-P', action='store_true', help='Print all ffmpeg output')

        return parser


@dataclass
class _ArgsMultiplier:
    num_segments: int
    args: argparse.Namespace

    def multiply_arg(self, arg_name: str, default: Any, index: int) -> Any:
        if not (arg := getattr(self.args, arg_name.replace('-', '_'))):
            return default

        if len(arg) == self.num_segments:
            return arg[index]

        if len(arg) == 1:
            return arg[0]

        terminate(f'Argument "{arg_name}" must be specified either for every segment or once')
