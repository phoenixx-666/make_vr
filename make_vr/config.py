import argparse
from dataclasses import dataclass, field


__all__ = ['Config', 'duration']


_DEFAULT_USYNC = 5


def duration(s: str | float | int) -> float:
    if isinstance(s, (float, int)):
        return float(s)
    return sum(float(seg) * (60.0 ** i) for i, seg in enumerate(reversed(s.split(':'))))


@dataclass
class Config:
    left: str | list[str]
    right: str | list[str]
    output: str | None

    threads: int
    get_offset: bool | None
    yes: bool | None
    overwrite: bool | None
    rename: bool | None
    print: bool | None

    ih_fov: float
    iv_fov: float

    fade: list[float]
    audio: int
    do_audio: bool

    bitrate: str
    duration: float | None
    preset: str
    extra_offset: float | None
    override_offset: float | None
    trim: float | None
    ultra_sync: int | None
    extra_video_filter: list[str] = field(default_factory=list)
    separate_audio: bool = False
    fill_end: bool = False

    quality: int = 2
    do_image: bool | None = None

    ffmpeg_path: str = ''
    ffprobe_path: str = ''
    ffmpeg_verbose: bool = False

    @staticmethod
    def from_args() -> 'Config':
        parser = argparse.ArgumentParser()
        parser.description = 'VR Video maker designed for processing GoPro fisheye videos'
        parser.add_argument('-l', '--left', type=str, nargs='+', metavar='path', required=True)
        parser.add_argument('-r', '--right', type=str, nargs='+', metavar='path', required=True)
        parser.add_argument('-o', '--output', type=str, metavar='path')

        parser.add_argument('-t', '--threads', type=int)
        parser.add_argument('--get-offset', action='store_true', help='Only calculate offset between files based on the audio tracks')
        parser.add_argument('-y', dest='yes', action='store_true', help='Answer "Yes" to all prompts')
        mex_group = parser.add_mutually_exclusive_group()
        mex_group.add_argument('-w', '--overwrite', action='store_true', help='Overwrite without prompt if output file already exists')
        mex_group.add_argument('--rename', action='store_true', help='Overwrite without prompt if output file already exists')
        parser.add_argument('-p', '--print', type=int, nargs='?', const=0, metavar='line_width', help='Only print ffmpeg command without executing it')

        arg_group = parser.add_argument_group('Camera options')
        arg_group.add_argument('--ihfov', type=float, default=122.0, help='Horizontal FOV of an input fisheye image')
        arg_group.add_argument('--ivfov', type=float, default=108.0, help='Vertical FOV of an input fisheye image')

        arg_group = parser.add_argument_group('Video options')
        arg_group.add_argument('-f', '--fade', type=duration, nargs='+', default=[1.0], help='Add fade effect in the beginning and the end of the video')
        arg_group.add_argument('-a', '--audio', type=int, choices=[-1, 0, 1], default=0, help='Use audio from the video file for the given eye, or -1 for no audio')
        arg_group.add_argument('-b', '--bitrate', type=str, help='Output video bitrate (Value must be compatible with ffmpeg)')
        arg_group.add_argument('-d', '--duration', type=duration, help='Maximum duration of the output video [seconds or h:m:s]')
        arg_group.add_argument('--preset', type=str, default='slow', help='Encoding preset (must be suitable for FFMpeg, default: slow)')
        arg_group.add_argument('--offset', type=duration, default=0.0, help='Extra offset for the cut video [seconds or h:m:s]')
        arg_group.add_argument('--override-offset', nargs=2, metavar=('INPUT', 'OFFSET'), help='Override offset on specified input')
        arg_group.add_argument('--trim', '-T', type=duration, default=0.0, help='Time to trim from the beginning, after the videos are synchronized')
        arg_group.add_argument('--ultra-sync', '-u', type=int, nargs='?', const=_DEFAULT_USYNC,
                                help=f'Apply extra synchronization by multiplying FPS by given number [default: {_DEFAULT_USYNC}], '
                                     'with interpolation and then lowering FPS back after trimming')
        arg_group.add_argument('--extra-video-filter', '-F', type=str, nargs='+', metavar='filter',
                                help='Custom video filters, applied before conversion from fisheye to equirectangular')
        arg_group.add_argument('--separate-audio', '-A', action='store_true', help='Store audio as a separate wav file for further editing')
        arg_group.add_argument('--fill-end', action='store_true',
                               help='If two videos are of unequal length after synchronization, '
                                    'fill the end of the resulting video with non-stereo content, '
                                    'up to the duration of the longer video')

        arg_group = parser.add_argument_group('Photo options')
        arg_group.add_argument('-q', '--quality', type=int, default=2, help='Output photo quality')

        arg_group = parser.add_argument_group('Other options')
        arg_group.add_argument('--ffmpeg-path', type=str, default='ffmpeg', help='Path to the ffmpeg executable')
        arg_group.add_argument('--ffprobe-path', type=str, default='ffprobe', help='Path to the ffprobe executable')
        arg_group.add_argument('--ffmpeg-print', '-P', action='store_true', help='Print all ffmpeg output')

        args = parser.parse_args()

        return Config(
            left=args.left,
            right=args.right,
            output=args.output,

            threads=args.threads,
            get_offset=args.get_offset,
            yes=args.yes,
            overwrite=args.overwrite,
            rename=args.rename,
            print=args.print,

            ih_fov=args.ihfov,
            iv_fov=args.ivfov,

            fade=args.fade,
            audio=args.audio,
            do_audio=args.audio != -1,
            bitrate=args.bitrate,
            duration=args.duration,
            preset=args.preset,
            extra_offset=args.offset,
            override_offset=args.override_offset,
            trim=args.trim,
            ultra_sync=args.ultra_sync,
            extra_video_filter=args.extra_video_filter or [],
            separate_audio=args.separate_audio,
            fill_end=args.fill_end,

            quality=args.quality,

            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
            ffmpeg_verbose=args.ffmpeg_print,
        )
