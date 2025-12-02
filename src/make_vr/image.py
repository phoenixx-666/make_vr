import subprocess as sp

from .task import Task
from .filters import Filter, FilterSeq, FilterGraph
from .fs import get_output_filename
from .shell import FFMpegCommand


__all__ = ['make_image']


SIDE = 4872
O_FOV = 180


def make_image(task: Task):
    (ffmpeg_command := FFMpegCommand(task.ffmpeg_path)).outputs.append(FFMpegCommand.Output(
        mappings=['-map',  '[photo]'],
        codecs=['-q', f'{task.quality}'],
        outputs=['-vframes', '1', '-update', '1', get_output_filename(task)]
    ))

    if task.quality == 1:
        ffmpeg_command.outputs[0].codecs.extend(['-qmin', '1'])

    if task.threads:
        ffmpeg_command.general_params.extend(['-threads', str(task.threads)])
    if task.overwrite or not task.do_print:
        ffmpeg_command.general_params.extend(['-y'])

    ffmpeg_command.inputs.extend([
        ['-f', 'lavfi', '-i', Filter('color',
                                     color='black',
                                     s=f'{(task.ow or SIDE) * 2}x{task.oh or SIDE}').render()],
        ['-i', (segment := task.segments[0]).left[0]],
        ['-i', segment.right[0]],
    ])

    filters = [Filter(filter_str) for filter_str in segment.extra_video_filter]
    filters.append(Filter('v360', 'fisheye', task.of, 'lanc',
                          iv_fov=segment.iv_fov,
                          ih_fov=segment.ih_fov,
                          h_fov=task.oh_fov,
                          v_fov=task.ov_fov,
                          alpha_mask=1,
                          w=task.ow or SIDE,
                          h=task.oh or SIDE))
    ffmpeg_command.filter_graph = FilterGraph([
        FilterSeq(['1:v:0'], ['left'], filters),
        FilterSeq(['2:v:0'], ['right'], filters),
        FilterSeq(['left', 'right'], ['overlay'], [Filter('hstack', inputs=2)]),
        FilterSeq(['0:v:0', 'overlay'], ['photo'], [Filter('overlay')]),
    ])

    if task.do_print:
        ffmpeg_command.print()
        exit()

    sp.run(ffmpeg_command.as_list(), shell=True)
