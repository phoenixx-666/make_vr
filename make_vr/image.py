import subprocess as sp

from .config import Config
from .filters import Filter, FilterSeq, FilterGraph
from .fs import get_output_filename
from .shell import FFMpegCommand


__all__ = ['make_image']


SIDE = 4872
O_FOV = 180


def make_image(cfg: Config):
    (ffmpeg_command := FFMpegCommand(cfg.ffmpeg_path)).outputs.append(FFMpegCommand.Output(
        mappings=['-map',  '[photo]'],
        codecs=['-q', f'{cfg.quality}'],
        outputs=['-vframes', '1', '-update', '1', get_output_filename(cfg)]
    ))

    if cfg.threads:
        ffmpeg_command.general_params.extend(['-threads', str(cfg.threads)])
    if cfg.overwrite or not cfg.do_print:
        ffmpeg_command.general_params.extend(['-y'])

    ffmpeg_command.inputs.extend([
        ['-f', 'lavfi', '-i', Filter('color', color='black', s=f'{SIDE * 2}x{SIDE}').render()],
        ['-i', (segment := cfg.segments[0]).left[0]],
        ['-i', segment.right[0]],
    ])

    filters = [Filter(filter_str) for filter_str in segment.extra_video_filter]
    filters.append(Filter('v360', 'fisheye', 'e', 'lanc', iv_fov=segment.iv_fov, ih_fov=segment.ih_fov,
                          h_fov=O_FOV, v_fov=O_FOV, alpha_mask=1, w=SIDE, h=SIDE))
    ffmpeg_command.filter_graph = FilterGraph([
        FilterSeq(['1:v:0'], ['left'], filters),
        FilterSeq(['2:v:0'], ['right'], filters),
        FilterSeq(['left', 'right'], ['overlay'], [Filter('hstack', inputs=2)]),
        FilterSeq(['0:v:0', 'overlay'], ['photo'], [Filter('overlay')]),
    ])

    if cfg.do_print:
        ffmpeg_command.print()
        exit()

    sp.run(ffmpeg_command.as_list(), shell=True)
