import itertools
from msvcrt import getch
import os

import filetype

from .config import Config


__all__ = ['get_output_filename', 'swap_extension', 'resolve_existsing', 'find_closest', 'validate_input_files']


def find_closest(dir: str, reference: str) -> tuple[str | None, float]:
    ref_edit_date = os.stat(reference).st_mtime_ns
    ref_ext = os.path.splitext(reference)[1].lower()
    diff = float('inf')
    closest = None
    for file in (f for f in next(os.walk(dir))[2] if os.path.splitext(f)[1].lower() == ref_ext):
        if (newdiff := abs(os.stat(fn := os.path.join(dir, file)).st_mtime_ns - ref_edit_date)) < diff:
            closest, diff = fn, newdiff
    return closest, diff / 1e9


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
            closest, diff = find_closest(left[0], right[0])
            left = [closest]
        else:
            closest, diff = find_closest(right[0], left[0])
            right = [closest]
        if closest:
            if cfg.yes:
                print(f'Found file "{closest}" with difference of {diff:.3f} seconds.')
            else:
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
            ft = "image" if (do_image := filetype.is_image(left[0] or right[0])) else "video"
            print(f'Unable to find corresponding {ft}')
            exit(1)

    if do_image is None:
        do_image = filetype.is_image(left[0])

    cfg.left, cfg.right, cfg.do_image = left, right, do_image
