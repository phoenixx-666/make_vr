import io
import math
import subprocess as sp

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal
from tqdm import tqdm

from .config import Config
from .filters import Filter, FilterSeq, FilterGraph, fts
from .shell import FFMpegCommand


__all__ = ['get_samples', 'find_offset']


def get_samples(cfg: Config, segment: Config.Segment, param: str, duration: float, sample_rate: int) -> ArrayLike:
    desc = f'Extracting samples ({param[0].upper()} eye)'
    filenames = getattr(segment, param)
    command = FFMpegCommand(cfg.ffmpeg_path)
    if cfg.threads:
         command.general_params.extend(['-threads', str(cfg.threads)])
    command.general_params.extend(['-hide_banner', '-loglevel', 'warning'])
    for fn in filenames:
        command.inputs.append(['-i', fn])
    if len(filenames) > 1:
        command.filter_graph = FilterGraph([
            FilterSeq([f'{i}:a:0' for i in range(len(filenames))], ['out'], [Filter('concat', n=len(filenames), v=0, a=1)])
        ])
        command.codecs_and_outputs.extend(['-map', '[out]'])
    else:
        command.codecs_and_outputs.extend(['-map', '0:a:0'])
    command.codecs_and_outputs.extend(['-vn', '-c:a', 'pcm_s16le', '-ac', '1', '-t', fts(duration), '-f', 's16le', 'pipe:1'])

    byte_count = math.ceil(duration * sample_rate) * np.dtype(np.int16).itemsize

    with (sp.Popen(command.as_list(), stdout=sp.PIPE) as proc,
          tqdm(total=byte_count, unit='B', unit_scale=True, unit_divisor=1024, desc=desc) as pbar):
        bio = io.BytesIO()

        while bts := proc.stdout.read(0x10000):
            bio.write(bts)
            pbar.update(len(bts))

        if bio.tell() < byte_count:
            pbar.update(byte_count - bio.tell())

        return np.frombuffer(bio.getbuffer(), np.int16).astype(np.float64)


def find_offset(audio1: ArrayLike, audio2: ArrayLike, rate: int) -> float:
    print(f'Calculating offset between two videos...', end='')
    correlation = signal.correlate(audio2, audio1, mode='full')
    lags = signal.correlation_lags(audio2.size, audio1.size, mode='full')
    lag = lags[np.argmax(correlation)] / rate
    print('Done.')
    return lag
