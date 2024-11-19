import subprocess as sp
from typing import IO

import numpy as np
from numpy.typing import ArrayLike
from pydub import AudioSegment
from scipy import signal

from .config import Config
from .filters import Filter, FilterSeq, FilterGraph, fts


__all__ = ['get_wav_samples', 'find_offset']


def _read_audio(file: IO) -> tuple[ArrayLike, int]:
    segment = AudioSegment.from_file(file)
    data = np.array(segment.get_array_of_samples())

    if segment.channels > 1:
        data = data.reshape((-1, segment.channels)).mean(axis=1)
    rate = segment.frame_rate
    return data, rate


def get_wav_samples(cfg: Config, param: str, duration: float) -> tuple[ArrayLike, int]:
    print(f'Getting wav samples from {param} eye video...', end='')
    filenames = getattr(cfg, param)
    command = [cfg.ffmpeg_path]
    if cfg.threads:
         command.extend(['-threads', str(cfg.threads)])
    command.extend(['-hide_banner', '-loglevel', 'warning'])
    for fn in filenames:
        command.extend(['-i', fn])
    if len(filenames) > 1:
        filter_graph = FilterGraph([
            FilterSeq([f'{i}:a' for i in range(len(filenames))], ['out'], [Filter('concat', n=len(filenames), v=0, a=1)])
        ])
        command.extend(['-filter_complex', filter_graph.render()])
        command.extend(['-map', '[out]'])
    else:
        command.extend(['-map', '0:a'])
    command.extend(['-vn', '-c:a', 'pcm_s16le',
                    '-t', fts(duration), '-f', 'wav', 'pipe:1'])
    with sp.Popen(command, stdout=sp.PIPE) as proc:
        result = _read_audio(proc.stdout)
    print('Done.')
    return result


def find_offset(audio1: ArrayLike, audio2: ArrayLike, rate: int) -> float:
    print(f'Calculating offset between two videos...', end='')
    correlation = signal.correlate(audio2, audio1, mode='full')
    lags = signal.correlation_lags(audio2.size, audio1.size, mode='full')
    lag = lags[np.argmax(correlation)] / rate
    print('Done.')
    return lag
