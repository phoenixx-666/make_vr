from collections import deque
from dataclasses import dataclass
import datetime
from fractions import Fraction
import json
import math
import re
import subprocess as sp
import sys
from typing import Any

import pexpect
import pexpect.popen_spawn
from tqdm import tqdm

from .audio import get_wav_samples, find_offset
from .config import Config
from .filters import Filter, FilterSeq, FilterGraph, fts
from .fs import get_output_filename, resolve_existing, swap_extension
from .shell import print_command


__all__ = ['make_video']


SIDE = 4096
O_FOV = 180


@dataclass
class Metadata:
    fps: Fraction
    # timecode: tuple[int, int, int, int] | None
    duration: float


class PyTaskbarStub:
    @staticmethod
    def init():
        pass

    @staticmethod
    def setProgress(_):
        pass

    @staticmethod
    def setState(_):
        pass


def d_to_hms(d: float) -> str:
    h = int(d // 3600)
    m = int((d % 3600) // 60)
    s = f'{float(d) % 60:05.2f}'
    if h:
        return f'{h:02d}:{m:02d}:{s}'
    return f'{m:02d}:{s}'


def get_metadata(metadata: dict[str, Any]) -> Metadata:
    streams = metadata['streams']
    s0md = streams[0]
    timecode = re.split(r'[:;]', (s0md.get('tags', {}).get('timecode', '') or s0md.get('tags', {}).get('TIMECODE', '')))

    return Metadata(
        fps = Fraction(s0md['r_frame_rate']),
        # timecode = tuple(map(int, timecode)) if timecode[0] else None,
        duration = float(s0md.get('duration', 0)) or sum([60.0 ** i * float(x) for i, x in enumerate(s0md['tags']['DURATION'].split(':')[::-1])])
    )


def make_video(cfg: Config):
    try:
        import tzlocal
        tz = tzlocal.get_localzone()
        datetime_fmt = '%d %b %Y %H:%M:%S'
        time_fmt = '%H:%M:%S'
    except ImportError:
        tz = datetime.timezone.utc
        datetime_fmt = '%d %b %Y %H:%M:%S UTC'
        time_fmt = '%H:%M:%S UTC'

    left = cfg.left
    right = cfg.right
    out = get_output_filename(cfg)

    if cfg.do_audio and cfg.separate_audio:
        out_wav = resolve_existing(cfg, swap_extension(out, 'wav'))

    def probe(fn: str) -> Any:
        command = [cfg.ffprobe_path, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', fn]
        with sp.Popen(command, stdout=sp.PIPE) as proc:
            return json.load(proc.stdout)

    metadata = [list(map(get_metadata, map(probe, inputs)))
                for inputs in (left, right)]
    if cfg.override_offset:
        cut_index = int(cfg.override_offset[0])
        time_diff = duration(cfg.override_offset[1])
    else:
        if cfg.wav_duration is not None:
            wav_duration = cfg.wav_duration
        else:
            wav_duration = abs(sum(map(lambda m: m.duration, metadata[0])) - sum(map(lambda m: m.duration, metadata[1]))) + 60.0
        print(f'wav_duration={d_to_hms(wav_duration)} ({fts(wav_duration)} s)')
        left_a, rate = get_wav_samples(cfg, 'left', wav_duration)
        right_a, _ = get_wav_samples(cfg, 'right', wav_duration)
        time_diff = find_offset(left_a, right_a, rate)
        if time_diff < 0.0:
            time_diff = -time_diff
            cut_index = 0
        elif time_diff > 0.0:
            cut_index = 1
        else:
            cut_index = -1
        if cfg.get_offset:
            print(f'Offset is {d_to_hms(time_diff)} (({fts(time_diff)} s), {cut_index})')
            exit()

    if cfg.extra_offset:
        time_diff += cfg.extra_offset

    durations = tuple(sum(m.duration for m in metadatas) - cfg.trim - (time_diff if i == cut_index else 0)
                      for i, metadatas in enumerate(metadata))
    duration = (max if cfg.fill_end else min)(durations)
    if cfg.duration and cfg.duration < duration:
        duration = cfg.duration
        if cfg.fill_end and duration <= min(durations):
            cfg.fill_end = False

    print(f'time_diff={d_to_hms(time_diff)} ({fts(time_diff)} s)')
    print(f'cut_index={cut_index:d}')
    print(f'duration={d_to_hms(duration)} ({fts(duration)} s)')

    command = [cfg.ffmpeg_path]
    if cfg.threads:
        command.extend(['-threads', str(cfg.threads)])
    command.extend(['-y'])

    fps = metadata[0][0].fps
    command.extend(['-f', 'lavfi', '-i', Filter('color', color='black', s=f'{SIDE * 2}x{SIDE}', r=fps).render()])

    all_inputs = (left_inputs, right_inputs) = [], []
    k = 1
    for i in range(2):
        for input in [cfg.left, cfg.right][i]:
            command.extend(['-i', input])
            all_inputs[i].append(k)
            k += 1

    all_input_str = left_input_str, right_input_str = [[f'{k}:v' for k in inputs] for inputs in (left_inputs, right_inputs)]

    print(all_inputs)

    all_filters = left_filters, right_filters = [], []
    all_outputs = left_outputs, right_outputs = ['left_raw'], ['right_raw']
    for i in range(2):
        all_filters[i].append(Filter('concat', n=len(all_inputs[i]), v=1, a=0))
        offset = cfg.trim
        if cut_index == i:
            offset += time_diff
        if offset > 0.0:
            if cfg.ultra_sync:
                new_fps = fps * cfg.ultra_sync
                # all_filters[i].append(Filter('minterpolate', fps=new_fps, mi_mode='mci'))
                all_filters[i].append(Filter('framerate', fps=new_fps))
            all_filters[i].extend([Filter('trim', start=offset), Filter('setpts', f'PTS-STARTPTS')])
            if cfg.ultra_sync:
                # all_filters[i].append(Filter('minterpolate', fps=fps))
                all_filters[i].append(Filter('framerate', fps=fps))

        if cfg.fill_end:
            if duration == durations[i]:
                all_filters[i].append(Filter('split'))
                all_outputs[i].append('filler_raw')
            else:
                all_input_str[i].append('filler')
                all_filters[i][0].kw_params['n'] += 1

    v360 = Filter('v360', 'fisheye', 'e', 'lanc', iv_fov=cfg.iv_fov, ih_fov=cfg.ih_fov, h_fov=O_FOV, v_fov=O_FOV, alpha_mask=1, w=SIDE, h=SIDE)
    filters = [Filter(filter_str) for filter_str in cfg.extra_video_filter] + [v360]

    filter_seqs = [
        FilterSeq(left_input_str, left_outputs, left_filters),
        FilterSeq(right_input_str, right_outputs, right_filters),
    ]
    if cfg.fill_end:
        filter_seqs.insert(1, FilterSeq(['filler_raw'], ['filler'], [
            Filter('trim', start=min(durations)),
            Filter('setpts', f'PTS-STARTPTS'),
        ]))
        if duration == durations[1]:
            filter_seqs.reverse()
    filter_seqs.extend([
        FilterSeq(['left_raw'], ['left'], filters),
        FilterSeq(['right_raw'], ['right'], filters),
        FilterSeq(['left', 'right'], ['overlay'], [Filter('hstack', inputs=2)]),
        video_fs := FilterSeq(['0:v', 'overlay'], ['video'], [Filter('overlay')])
    ])
    filter_graph = FilterGraph(filter_seqs)

    fade_in = cfg.fade[0]
    fade_out = cfg.fade[1] if len(cfg.fade) >= 2 else cfg.fade[0]

    if fade_in:
        video_fs.filters.append(Filter('fade', t='in', st=0, d=fade_in))
    if fade_out:
        video_fs.filters.append(Filter('fade', t='out', st=duration - fade_out, d=fade_out))
    video_fs.filters.append(Filter('trim', duration=duration))

    if cfg.do_audio:
        audio_inputs = [f'{i}:a' for i in ([left_inputs, right_inputs][cfg.audio])]
        audio_filters = [Filter('concat', n=len(audio_inputs), v=0, a=1)]
        offset = cfg.trim
        if cut_index == cfg.audio:
            offset += time_diff
        if offset > 0.0:
            audio_filters.extend([Filter('atrim', start=offset), Filter('asetpts', f'PTS-STARTPTS')])

        if fade_in:
            audio_filters.append(Filter('afade', t='in', st=0, d=fade_in))
        if fade_out:
            audio_filters.append(Filter('afade', t='out', st=duration - fade_out, d=fade_out))
        audio_filters.append(Filter('atrim', duration=duration))
        filter_graph.filter_seqs.append(FilterSeq(audio_inputs, ['audio'], audio_filters))

    command.extend(['-filter_complex', filter_graph.render()])

    command.extend(['-map', '[video]'])
    if cfg.do_audio and not cfg.separate_audio:
        command.extend(['-map', '[audio]'])
    command.extend(['-c:v', 'hevc_nvenc'])
    if cfg.bitrate:
        command.extend(['-vb', cfg.bitrate])

    command.extend(['-preset', cfg.preset])
    if cfg.do_audio and not cfg.separate_audio:
        command.extend(['-c:a', 'aac', '-ab', '192k'])
    command.extend(['-t', fts(duration)])
    command.extend([out])

    if cfg.do_audio and cfg.separate_audio:
        command.extend(['-map', '[audio]'])
        command.extend(['-c:a', 'pcm_s16le'])
        command.extend(['-t', fts(duration)])
        command.extend([out_wav])

    if cfg.print is not None:
        print_command(cfg, command)

    num_frames = round(float(duration * fps))
    print(f'num_frames={num_frames}')
    print(f'Output Filename: "{out}"')
    proc = None
    pbar = None

    if sys.platform == 'win32':
        try:
            import PyTaskbar
            task_prog = PyTaskbar.Progress()
            task_prog.init()
        except ImportError:
            task_prog = PyTaskbarStub()
    else:
        task_prog = PyTaskbarStub()

    try:
        if sys.platform.startswith('win'):
            proc = pexpect.popen_spawn.PopenSpawn(command)
        else:
            proc = pexpect.spawn(command)
        cpl = proc.compile_pattern_list([
            pexpect.EOF,
            r'frame= *(\d+).*',
            r'(.+)'
        ])
        last_frame = 0
        timestamps: deque[tuple[datetime.datetime, int]] = deque()
        while True:
            i = proc.expect_list(cpl, timeout=None)
            if i == 0:  # EOF
                break
            elif i == 1:
                if not pbar:
                    pbar = tqdm(total=num_frames, smoothing=0.1, unit='f',
                                bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, f={n_fmt}, {rate_fmt}{postfix}]',
                                postfix=f't={d_to_hms(0)}, x?, eta=?')
                    task_prog.setState('loading')
                    task_prog.setProgress(0)
                cur_frame = int(proc.match.group(1))
                if cur_frame > last_frame:
                    now = datetime.datetime.now()
                    while len(timestamps) > 1 and (now - timestamps[0][0]).total_seconds() > 10.0:
                        timestamps.popleft()
                    timestamps.append((now, cur_frame))
                    if len(timestamps) >= 2 and ((time_diff := (now - timestamps[0][0]).total_seconds()) > 0.0):
                        prog_fps = (cur_frame - timestamps[0][1]) / time_diff
                        ratio = float(prog_fps / fps)
                        if (len_frac := 4 - len(str(int(ratio)))) > 0:
                            mul = f'{{0:0.{len_frac}f}}'.format(ratio)
                        else:
                            mul = int(ratio)

                        if prog_fps != 0.0:
                            remaining_frames = num_frames - cur_frame
                            now = datetime.datetime.now(tz)
                            eta_dt = datetime.datetime.now(tz) + datetime.timedelta(seconds=remaining_frames / prog_fps)
                            eta = eta_dt.strftime(time_fmt if eta_dt.date() == now.date() else datetime_fmt)
                        else:
                            eta = '?'
                    else:
                        mul = eta = '?'

                    pbar.set_postfix_str(f't={d_to_hms(float(cur_frame / fps))}, x{mul}, eta={eta}')
                    pbar.update(cur_frame - last_frame)
                    task_prog.setProgress(round(cur_frame / (num_frames - 1) * 100))
                    last_frame = cur_frame
                if hasattr(proc, 'close'):
                    proc.close
            elif i == 2:
                if cfg.ffmpeg_verbose:
                    tqdm.write(proc.match.group(0).decode(), end='')
        pbar and pbar.update(num_frames - last_frame)
    except KeyboardInterrupt:
        if proc:
            print('Progress was interrupted!')
            import signal
            proc.kill(signal.SIGTERM)
            task_prog.setState('normal')
    else:
        task_prog.setState('done')
    finally:
        pbar and pbar.close()
        task_prog.setProgress(0)
        print(f'Program finished at {datetime.datetime.now(tz).strftime(datetime_fmt)}')
