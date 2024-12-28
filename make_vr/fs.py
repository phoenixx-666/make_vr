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

from .config import Config


__all__ = ['probe', 'get_output_filename', 'swap_extension', 'resolve_existsing', 'find_closest', 'validate_input_files']


def probe(cfg: Config, fn: str) -> Any:
    command = [cfg.ffprobe_path, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', fn]
    with sp.Popen(command, stdout=sp.PIPE) as proc:
        return json.load(proc.stdout)


def get_creation_date(cfg: Config, fn: str) -> datetime:
    json_data = probe(cfg, fn)
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


def find_closest(cfg: Config, dir: str, reference: str) -> tuple[str | None, float]:
    diff = float('inf')
    closest = None

    ref_date = get_creation_date(cfg, reference)
    ref_ext = os.path.splitext(reference)[1].lower()

    for file in (f for f in next(os.walk(dir))[2] if os.path.splitext(f)[1].lower() == ref_ext):
        fn = os.path.join(dir, file)
        date = get_creation_date(cfg, fn)
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
        elif not (cfg.yes or cfg.overwrite):
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
        multiple = False

        if len(cfg.left) > 1:
            ext = os.path.splitext(os.path.basename(cfg.left[0]))[1]
            basename = '_'.join(map(lambda fn: os.path.splitext(os.path.basename(fn))[0], cfg.left))
            multiple = True
        else:
            basename, ext = os.path.splitext(os.path.basename(cfg.left[0]))

        if len(cfg.right) > 1:
            basename2 = '_'.join(map(lambda fn: os.path.splitext(os.path.basename(fn))[0], cfg.right))
            multiple = True
        else:
            basename2 = os.path.splitext(os.path.basename(cfg.right[0]))[0]

        if cfg.do_stab:
            ext = '.trf'

        if multiple:
            fn = f'{basename}__{basename2}{ext.lower()}'
        else:
            fn = f'{basename}_{basename2}{ext.lower()}'

    return resolve_existing(cfg, fn)


def validate_input_files(cfg: Config):
    left = list(map(os.path.normpath, cfg.left))
    right = list(map(os.path.normpath, cfg.right))

    for fn in itertools.chain(left, right):
        if not os.path.exists(fn):
            print(f'Input path {fn} does not exist!')
            exit(1)

    left_dir = any(map(os.path.isdir, left))
    right_dir = any(map(os.path.isdir, right))

    do_image = None

    if any([left_dir, right_dir]):
        if len(left) > 1 or len(right) > 1:
            print('If one of the inputs is a directory, only one name can be specified per input')
            exit(1)
        if all([left_dir, right_dir]):
            print('Both inputs can\'t be directories')
            exit(1)
        if left_dir:
            closest, diff = find_closest(cfg, left[0], right[0])
            left = [closest]
        else:
            closest, diff = find_closest(cfg, right[0], left[0])
            right = [closest]
        if closest:
            if cfg.ask_match:
                print(f'Found file "{closest}" with difference of {diff:.3f} seconds. Continue? [Y/n]', end=None)
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
                        ...
            else:
                print(f'Found file "{closest}" with difference of {diff:.3f} seconds.')

        else:
            ft = "image" if (do_image := filetype.is_image(left[0] or right[0])) else "video"
            print(f'Unable to find corresponding {ft}')
            exit(1)

    if do_image is None:
        do_image = filetype.is_image(left[0])

    cfg.left, cfg.right, cfg.do_image = left, right, do_image
