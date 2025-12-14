from dataclasses import dataclass
from datetime import datetime
from exif import Image
import filetype
import itertools
import json
from msvcrt import getch
import os
from pathlib import Path
import subprocess as sp
from tqdm import tqdm
from typing import Any

from .cache import Cache
from .config import Config
from .shell import CLIArgs, info, success, terminate
from .tools import with_each

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .task import Task


__all__ = ['Input', 'probe', 'get_modified_date', 'get_output_filename', 'swap_extension',
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


def get_modified_date(fn: str) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(fn))


def get_creation_date(ffprobe_path: str, fn: str) -> datetime | None:
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


def get_creation_date_cached(fn: str) -> datetime | None:
    if Cache().is_ignoring('match'):
        return get_creation_date(CLIArgs().ffprobe_path, fn)

    mod_date = get_modified_date(fn)
    if (Cache().get_modified(fn) == mod_date) and ((cache_create_date := Cache().get_match(fn)) is not None):
        creation_date = cache_create_date
    else:
        creation_date = get_creation_date(CLIArgs().ffprobe_path, fn)
        Cache().set_match(fn, mod_date, creation_date)
    return creation_date


def rename(fn: str) -> str:
    basename, ext = os.path.splitext(fn)
    counter = itertools.count(1)
    while os.path.exists(res := f'{basename}_{next(counter)}{ext}'):
        ...
    return res


def resolve_existing(task: Task, fn: str) -> str:
    if os.path.exists(fn):
        if task.rename:
            fn = rename(fn)
        elif not (task.do_print or task.overwrite):
            info(f'File "{fn}" already exists. Overwrite? [y/N/r]', end=None)
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


def get_output_filename(task: Task) -> str:
    if task.output:
        fn = os.path.normpath(task.output)
    else:
        ext = None
        fn_parts = []

        if task.do_stab:
            ext = '.trf'

        for left, right in ((segment.left, segment.right) for segment in task.segments):
            if ext is None:
                ext = os.path.splitext(os.path.basename(left[0]))[1]

            basename_l = '_'.join(map(lambda fn: os.path.splitext(os.path.basename(fn))[0], left))
            basename_r = '_'.join(map(lambda fn: os.path.splitext(os.path.basename(fn))[0], right))
            multiple = (len(left) + len(right)) > 2

            fn_parts.append(f'{basename_l}{"__" if multiple else "_"}{basename_r}')

        fn = f'{"+".join(fn_parts)}{ext.lower()}'

    return resolve_existing(task, fn)


def validate_input_files() -> Input:
    left = [list(map(os.path.normpath, left_segment)) for left_segment in (CLIArgs().left or [])]
    right = [list(map(os.path.normpath, right_segment)) for right_segment in (CLIArgs().right or [])]
    audio = [list(map(os.path.normpath, audio_segment)) for audio_segment in (CLIArgs().audio or [])]

    if not (left or right):
        terminate('Must specify at least one input')

    if not left:
        if not Config().inputs.left_dir:
            terminate('No input specified for the left eye')
        left = [[Config().inputs.left_dir]]
    elif not right:
        if not Config().inputs.right_dir:
            terminate('No input specified for the right eye')
        right = [[Config().inputs.right_dir]]

    do_stab = (CLIArgs().stab is not None) or (CLIArgs().stab_channel is not None)
    left_is_dir = right_is_dir = False

    if do_stab and not (len(left) == len(right) == 1):
        terminate('Stabilization estimation should only be performed on single-segment videos')

    if len(left) != len(right):
        left_is_dir = len(left) == 1 and len(left[0]) == 1 and os.path.isdir(left[0][0])
        right_is_dir = len(right) == 1 and len(right[0]) == 1 and os.path.isdir(right[0][0])
        if left_is_dir == right_is_dir:
            if left_is_dir:
                terminate('If one of the inputs is a directory, it must be single')
            else:
                terminate('If multiple segments are specified, the number of input sequences for left and right eye must be equal.\n'
                          'Or otherwise a single folder specified for the other eye.')

        if left_is_dir:
            left = [left[0].copy() for _ in range(len(right))]
        elif right_is_dir:
            right = [right[0].copy() for _ in range(len(left))]

    for fn in itertools.chain(*left, *right, *audio):
        if not os.path.exists(fn):
            terminate(f'Input path {fn} does not exist!')

    do_image = False

    dir_segment: dict[int, tuple[str, bool]] = {}
    dir_ext: dict[str, set[str]] = {}

    for i, (left_files, right_files) in enumerate(zip(left, right)):
        left_segment_is_dir = left_is_dir or os.path.isdir(left_files[0])
        right_segment_is_dir = right_is_dir or os.path.isdir(right_files[0])

        if any([left_segment_is_dir, right_segment_is_dir]):
            if len(left_files) > 1 or len(right_files) > 1:
                terminate('If a segment has a directory specified for one eye, other eye can have only one file')
            if all([left_segment_is_dir, right_segment_is_dir]):
                terminate("Both inputs can't be directories")
            if left_segment_is_dir:
                dir_ext.setdefault(left_files[0], set()).add(os.path.splitext(right_files[0])[1].lower())
                dir_segment[i] = (left_files[0], False)
            else:
                dir_ext.setdefault(right_files[0], set()).add(os.path.splitext(left_files[0])[1].lower())
                dir_segment[i] = (right_files[0], True)
        elif any(map(os.path.isdir, itertools.chain(left_files, right_files))):
            terminate("Can't mix files and directories in inputs")

        if do_image := ((not left_segment_is_dir) and filetype.is_image(left_files[0])) or \
                       ((not right_segment_is_dir) and filetype.is_image(right_files[0])):
            if ((not left_segment_is_dir) and (not filetype.is_image(left_files[0]))) or \
               ((not right_segment_is_dir) and (not filetype.is_image(right_files[0]))):
                terminate("Can't mix images and videos in inputs")

            if i or len(left_files) > 1 or len(right_files) > 1:
                terminate('Can generate only one image at a time')

    if audio:
        if len(audio) != len(left):
            terminate('If external audio is specified, the number of audio segments must match that of video segments!\n'
                      "In this case, to use audio embedded in video, specify --audio argument with no files.")
    else:
        audio = [[]] * len(left)

    if dir_segment:
        dir_files: dict[str, list[tuple[str, datetime]]] = {}
        for dir_name, right_is_dir in dir_segment.values():
            try:
                ext = dir_ext[dir_name]
                file_list = [(str(Path(dir_name) / fn), None) for fn in next(os.walk(dir_name))[2]
                             if os.path.splitext(fn)[1].lower() in ext]
            except StopIteration:
                terminate(f'Input directory "{dir_name}" does not exist!')
            if not file_list:
                terminate(f'Input directory "{dir_name}" does not contain suitable files!')
            dir_files[dir_name] = file_list

        pbar = tqdm(desc='Retrieving creation dates', total=sum(map(len, dir_files.values())),
                    bar_format='\033[93m{l_bar}{bar}{r_bar}\033[0m')
        for dir_name, (i, (file_name, _)) in itertools.chain.from_iterable(
                map(lambda fwd: with_each(fwd[0], enumerate(fwd[1])), dir_files.items())):
            creation_date = get_creation_date_cached(file_name)
            dir_files[dir_name][i] = (file_name, creation_date)
            pbar.update()
        pbar.close()

        for dir_name, file_list in dir_files.items():
            dir_files[dir_name] = list(filter(lambda item: item[1] is not None, file_list))

        for i, (dir_name, right_is_dir) in dir_segment.items():
            target = left[i][0] if right_is_dir else right[i][0]
            if not (target_date := get_creation_date_cached(target)):
                terminate(f'Unable to determine creation date for the file "{target}"')
            diff = float('inf')
            closest = None

            for file_name, date in dir_files[dir_name]:
                if (new_diff := abs((target_date - date).total_seconds())) < diff:
                    closest, diff = file_name, new_diff

            if closest:
                prompt = f'Found file "{closest}" matching "{target}" with difference of {diff:.3f} seconds.'
                if CLIArgs().ask_match:
                    success(f'{prompt} Continue? [Y/n]', end=None)
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
                    success(prompt)

                (right if right_is_dir else left)[i] = [closest]
            else:
                ft = 'image' if do_image else 'video'
                terminate(f'Unable to find corresponding {ft} for the file "{target}"')

        Cache().save_if_updated()

    return Input(do_image, do_stab, [Input.Segment(*lra) for lra in zip(left, right, audio)])
