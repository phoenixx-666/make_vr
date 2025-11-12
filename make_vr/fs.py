import argparse
from dataclasses import dataclass
from datetime import datetime
from exif import Image
import filetype
import itertools
import json
from msvcrt import getch
import os
import subprocess as sp
from typing import Any

import filetype

from .shell import terminate

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import Config


__all__ = ['Inputs', 'probe', 'get_output_filename', 'swap_extension',
           'resolve_existsing', 'find_closest', 'validate_input_files']


@dataclass
class Input:
    @dataclass
    class Segment:
        left: list[str]
        right: list[str]
        audio: list[str]

    do_image: bool
    do_stab: bool
    segments: list[Segment]


def probe(ffprobe_path: str, fn: str) -> Any:
    command = [ffprobe_path, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', fn]
    with sp.Popen(command, stdout=sp.PIPE) as proc:
        return json.load(proc.stdout)


def get_creation_date(ffprobe_path: str, fn: str) -> datetime:
    json_data = probe(ffprobe_path, fn)
    try:
        date = json_data['format']['tags']['creation_time']
        return datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ')
    except (TypeError, KeyError, ValueError, PermissionError):
        ...

    try:
        if filetype.is_image(fn):
            with open(fn, 'rb') as image_file:
                image = Image(image_file)
            if not image.has_exif:
                return None
            return datetime.strptime(image.datetime, '%Y:%m:%d %H:%M:%S')
    except (TypeError, AttributeError, ValueError, PermissionError):
        ...

    return None


def find_closest(ffprobe_path: str, dir: str, reference: str) -> tuple[str | None, float]:
    diff = float('inf')
    closest = None

    ref_date = get_creation_date(ffprobe_path, reference)
    ref_ext = os.path.splitext(reference)[1].lower()

    for file in (f for f in next(os.walk(dir))[2] if os.path.splitext(f)[1].lower() == ref_ext):
        fn = os.path.join(dir, file)
        date = get_creation_date(ffprobe_path, fn)
        if not date:
            continue
        newdiff = abs(date - ref_date).total_seconds()
        if newdiff < diff:
            closest, diff = fn, newdiff
    return closest, diff


def rename(fn: str) -> str:
    basename, ext = os.path.splitext(fn)
    counter = itertools.count(1)
    while os.path.exists(res := f'{basename}_{next(counter)}{ext}'):
        ...
    return res


def resolve_existing(cfg: Config, fn: str) -> str:
    if os.path.exists(fn):
        if cfg.rename:
            fn = rename(fn)
        elif not (cfg.do_print or cfg.overwrite):
            print(f'File "{fn}" already exists. Overwrite? [y/N/r]', end=None)
            while True:
                resp = getch()
                try:
                    resp = resp.decode().lower()
                    if resp not in 'ynr\r\n':
                        continue
                    elif resp == 'y':
                        break
                    elif resp == 'r':
                        fn = rename(fn)
                        break
                    else:
                        exit()
                except UnicodeDecodeError:
                    ...
    return fn


def swap_extension(fn: str, new_ext: str) -> str:
    if new_ext and not new_ext.startswith('.'):
        new_ext = f'.{new_ext}'
    return f'{os.path.splitext((fn))[0]}{new_ext}'


def get_output_filename(cfg: Config) -> str:
    if cfg.output:
        fn = os.path.normpath(cfg.output)
    else:

        ext = None

        if cfg.do_stab:
            ext = '.trf'

        fn_parts = []

        for left, right in ((segment.left, segment.right) for segment in cfg.segments):
            multiple = False

            if len(left) > 1:
                if ext is None:
                    ext = os.path.splitext(os.path.basename(left[0]))[1]
                basename = '_'.join(map(lambda fn: os.path.splitext(os.path.basename(fn))[0], left))
                multiple = True
            else:
                basename, ext = os.path.splitext(os.path.basename(left[0]))

            if len(right) > 1:
                basename2 = '_'.join(map(lambda fn: os.path.splitext(os.path.basename(fn))[0], right))
                multiple = True
            else:
                basename2 = os.path.splitext(os.path.basename(right[0]))[0]

            fn_parts.append(f'{basename}{"__" if multiple else "_"}{basename2}')

        fn = f'{"+".join(fn_parts)}{ext.lower()}'

    return resolve_existing(cfg, fn)


def validate_input_files(args: argparse.Namespace) -> Input:
    left = [list(map(os.path.normpath, left_segment)) for left_segment in (args.left or [])]
    right = [list(map(os.path.normpath, right_segment)) for right_segment in (args.right or [])]
    audio = [list(map(os.path.normpath, audio_segment)) for audio_segment in (args.audio or [])]

    do_stab = (args.stab is not None) or (args.stab_channel is not None)
    left_dir = right_dir = False

    if do_stab and not (len(left) == len(right) == 1):
        terminate('Stabilization estimation should only be performed on single-segment videos')

    if len(left) != len(right):
        left_dir = len(left) == 1 and len(left[0]) == 1 and os.path.isdir(left[0][0])
        right_dir = len(right) == 1 and len(right[0]) == 1 and os.path.isdir(right[0][0])
        if left_dir == right_dir:
            if left_dir:
                terminate('If one of the inputs is a directory, it must be single')
            else:
                terminate('If multiple segments are specified, the number of input sequences for left and right eye must be equal.\n'
                          'Or otherwise a single folder specified for the other eye.')

        if left_dir:
            left = [left[0].copy() for _ in range(len(right))]
        elif right_dir:
            right = [right[0].copy() for _ in range(len(left))]

    for fn in itertools.chain(*left, *right, *audio):
        if not os.path.exists(fn):
            terminate(f'Input path {fn} does not exist!')

    first = True
    do_image = False

    for left_files, right_files in zip(left, right):
        left_dir_segment = left_dir or os.path.isdir(left_files[0])
        right_dir_segment = right_dir or os.path.isdir(right_files[0])

        if any([left_dir_segment, right_dir_segment]):
            if len(left_files) > 1 or len(right_files) > 1:
                terminate('If a segment has a directory specified for one eye, other eye can have only one file')
            if all([left_dir_segment, right_dir_segment]):
                terminate("Both inputs can't be directories")
            if left_dir_segment:
                closest, diff = find_closest(args.ffprobe_path, left_files[0], target := right_files[0])
                left_files.clear()
                left_files.append(closest)
            else:
                closest, diff = find_closest(args.ffprobe_path, right_files[0], target := left_files[0])
                right_files.clear()
                right_files.append(closest)
            if closest:
                prompt = f'Found file "{closest}" matching "{target}" with difference of {diff:.3f} seconds.'
                if args.ask_match:
                    print(f'{prompt} Continue? [Y/n]', end=None)
                    while True:
                        resp = getch()
                        try:
                            resp = resp.decode().lower()
                            if resp not in 'yn\r\n':
                                continue
                            elif resp == 'n':
                                exit()
                            else:
                                break
                        except UnicodeDecodeError:
                            terminate('Bad unicode character input')
                else:
                    print(prompt)
            else:
                ft = "image" if (do_image := filetype.is_image(left[0] or right[0])) else "video"
                terminate(f'Unable to find corresponding {ft}')
        else:
            for fn in itertools.chain(left_files, right_files):
                if os.path.isdir(fn):
                    terminate("Can't mix files and directories in inputs")

        if do_image := filetype.is_image(left_files[0]):
            if not filetype.is_image(right_files[0]):
                terminate("Can't mix images and videos in inputs")

            if (not first) or len(left_files) > 1 or len(right_files) > 1:
                terminate('Can generate only one image at a time')

        first = False

    if audio:
        if len(audio) != len(left):
            terminate('If external audio is specified, the number of audio segments must match that of video segments!\n'
                      "In this case, to use audio embedded in video, specify --audio argument with no files.")
    else:
        audio = [[]] * len(left)

    return Input(do_image, do_stab, [Input.Segment(*lra) for lra in zip(left, right, audio)])
