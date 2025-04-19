from msvcrt import getch
import os
import subprocess as sp

from .config import Config
from .filters import Filter, FilterSeq, FilterGraph
from .fs import get_output_filename
from .shell import print_command


__all__ = ['make_image']


SIDE = 4872
O_FOV = 180


def make_image(cfg: Config):
    if len(cfg.left) > 1 or len(cfg.right) > 1:
        print(f'Photo must have only one input for each eye')
        exit(1)

    left, right = cfg.left[0], cfg.right[0]
    out = get_output_filename(cfg)

    command = [cfg.ffmpeg_path]
    if cfg.threads:
        command.extend(['-threads', str(cfg.threads)])
    command.extend(['-y'])
    command.extend(['-f', 'lavfi', '-i', f'color=color=black:s={SIDE * 2:d}x{SIDE:d}'])
    command.extend(['-i', left])
    command.extend(['-i', right])

    extra_fitlers = [Filter(filter_str) for filter_str in cfg.extra_video_filter]
    v360 = Filter('v360', 'fisheye', 'e', 'lanc', iv_fov=cfg.iv_fov, ih_fov=cfg.ih_fov, h_fov=O_FOV, v_fov=O_FOV, alpha_mask=1, w=SIDE, h=SIDE)
    filter_complpex = FilterGraph([
        FilterSeq(['1:v:0'], ['left'], extra_fitlers + [v360]),
        FilterSeq(['2:v:0'], ['right'], extra_fitlers + [v360]),
        FilterSeq(['left', 'right'], ['overlay'], [Filter('hstack', inputs=2)]),
        FilterSeq(['0:v:0', 'overlay'], ['photo'], [Filter('overlay')]),
    ])

    command.extend(['-filter_complex', filter_complpex.render()])
    command.extend(['-map',  '[photo]'])
    command.extend(['-q', f'{cfg.quality}'])
    command.extend(['-vframes', '1', '-update', '1'])

    command.extend([out])

    if cfg.print is not None:
        print_command(cfg, command)

    sp.run(command, shell=True)
