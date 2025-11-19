from collections import deque
from dataclasses import dataclass
import datetime
from fractions import Fraction
import os
import sys
from typing import Any

import pexpect
import pexpect.popen_spawn
from tqdm import tqdm

from .audio import get_samples, find_offset
from .config import Config
from .filters import Filter, FilterSeq, FilterGraph, fts
from .fs import probe, get_output_filename, resolve_existing, swap_extension
from .shell import FFMpegCommand, terminate


__all__ = ['make_video']


SIDE = 4096
O_FOV = 180


@dataclass
class Metadata:
    fps: Fraction
    duration: float
    sample_rate: int
    channel_layout: str

    def __init__(self, metadata: dict[str, Any]):
        try:
            streams = metadata['streams']
            stream_type = 'video'
            video_stream = next(filter(lambda stream: stream.get('codec_type') == stream_type, streams))
            stream_type = 'audio'
            audio_stream = next(filter(lambda stream: stream.get('codec_type') == stream_type, streams))

            self.fps = Fraction(video_stream['r_frame_rate'])
            self.duration = float(video_stream.get('duration', 0)) or \
                sum([60.0 ** i * float(x) for i, x in enumerate(video_stream['tags']['DURATION'].split(':')[::-1])])
            self.sample_rate = int(audio_stream['sample_rate'])
            self.channel_layout = audio_stream['channel_layout']
        except KeyError as exc:
            terminate(f'Error reading metadata: lack of key "{exc.args[0]}"')
        except StopIteration:
            terminate(f'Unable to find {stream_type} stream')
        except ZeroDivisionError:
            terminate('Incorrect frame rate value')


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

    out = get_output_filename(cfg)

    if any(segment.do_audio for segment in cfg.segments) and cfg.separate_audio:
        out_wav = resolve_existing(cfg, swap_extension(out, 'wav'))

    ffmpeg_command = FFMpegCommand(cfg.ffmpeg_path)
    if cfg.threads:
        ffmpeg_command.general_params.extend(['-threads', str(cfg.threads)])
    if cfg.overwrite or not cfg.do_print:
        ffmpeg_command.general_params.extend(['-y'])

    segments = cfg.segments

    metadata = [[list(map(Metadata, map(lambda fn: probe(cfg.ffprobe_path, fn), files))) for files in inputs]
                 for inputs in ((segment.left, segment.right) for segment in segments)]
    fps = (base_metadata := metadata[0][0][0]).fps
    channel_layout = base_metadata.channel_layout
    sample_rate = base_metadata.sample_rate
    total_duration = 0.0
    any_do_audio = any(segment.do_audio for segment in segments)

    input_index = 0
    filter_seqs = []

    terminal_width = os.get_terminal_size().columns - 1

    for segment_index, segment in enumerate(segments, start=1):

        if len(segments) > 1:
            text = f' SEGMENT {segment_index} '
            width = max(0, terminal_width - len(text))
            print(f'{"=" * (width // 2)}{text}{"=" * (width // 2 + width % 2)}')

        seg_md = metadata[segment_index - 1]
        d_l = sum(map(lambda m: m.duration, seg_md[0]))
        d_r = sum(map(lambda m: m.duration, seg_md[1]))

        if segment.override_offset:
            cut_index = int(segment.override_offset[0])
            time_diff = duration(segment.override_offset[1])
        else:
            if segment.wav_duration is not None:
                wav_duration = segment.wav_duration
            else:
                wav_duration = abs(d_l - d_r) + 60.0
            print(f'wav_duration={d_to_hms(wav_duration)} ({fts(wav_duration)} s)')
            left_a = get_samples(cfg, segment, 'left', wav_duration, sample_rate)
            right_a = get_samples(cfg, segment, 'right', wav_duration, sample_rate)
            time_diff = find_offset(left_a, right_a, sample_rate)
            if time_diff < 0.0:
                time_diff = -time_diff
                cut_index = 0
            elif time_diff > 0.0:
                cut_index = 1
            else:
                cut_index = -1

        if segment.extra_offset:
            time_diff += segment.extra_offset

        durations = tuple(duration - segment.trim - (time_diff if i == cut_index else 0)
                          for i, duration in enumerate((d_l, d_r)))
        duration = (max if segment.fill_end else min)(durations)

        if segment.duration and segment.duration < duration:
            duration = segment.duration
            if segment.fill_end and duration <= min(durations):
                segment.fill_end = False

        total_duration += duration

        print(f'time_diff={d_to_hms(time_diff)} ({fts(time_diff)} s)')
        print(f'cut_index={cut_index:d}')
        print(f'duration={d_to_hms(duration)} ({fts(duration)} s)')

        inputs = []
        inputs_str = []

        if cfg.do_stab:
            for k, input in enumerate((segment.left, segment.right)[segment.stab_channel], start=input_index):
                ffmpeg_command.inputs.append(['-i', input])
                inputs.append(k)
                inputs_str.append(f'{k}:v:0')

            print(inputs)

            filters = []

            filters.append(Filter('concat', n=len(inputs), v=1, a=0))

            offset = segment.trim
            if cut_index == segment.stab_channel:
                offset += time_diff
            if offset > 0.0:
                if segment.ultra_sync:
                    new_fps = fps * segment.ultra_sync
                    # filters.append(Filter('minterpolate', fps=new_fps, mi_mode='mci'))
                    filters.append(Filter('framerate', fps=new_fps))
                filters.extend([Filter('trim', start=offset), Filter('setpts', f'PTS-STARTPTS')])
                if segment.ultra_sync:
                    # filters.append(Filter('minterpolate', fps=fps))
                    filters.append(Filter('framerate', fps=fps))

            if segment.fill_end:
                ...

            filters.append(Filter('trim', duration=duration))
            filters.append(vsd := Filter('vidstabdetect', result=out, fileformat='ascii'))
            if segment.stab_args:
                vsd.add_raw(segment.stab_args)
            filter_seqs.append(FilterSeq(inputs_str, [], filters))

            input_index += len(inputs)
        else:
            suffix = '' if len(segments) == 1 else str(segment_index)
            ffmpeg_command.inputs.append(['-f', 'lavfi', '-i', Filter('color', color='black', s=f'{SIDE * 2}x{SIDE}', r=fps).render()])

            all_inputs = (left_inputs, right_inputs) = [], []
            audio_inputs = []
            k = input_index + 1
            for i in range(2):
                for input in (segment.left, segment.right)[i]:
                    ffmpeg_command.inputs.append(['-i', input])
                    all_inputs[i].append(k)
                    k += 1
            if segment.external_audio:
                for input in segment.audio:
                    ffmpeg_command.inputs.append(['-i', input])
                    audio_inputs.append(k)
                    k += 1

            all_input_str = left_input_str, right_input_str = [[f'{k}:v:0' for k in inputs] for inputs in (left_inputs, right_inputs)]

            # print(all_inputs)
            # print(all_input_str)

            all_filters = left_filters, right_filters = [], []
            all_outputs = left_outputs, right_outputs = [f'left_raw{suffix}'], [f'right_raw{suffix}']
            for i in range(2):
                all_filters[i].append(Filter('concat', n=len(all_inputs[i]), v=1, a=0))
                offset = segment.trim
                if cut_index == i:
                    offset += time_diff
                if offset > 0.0:
                    if segment.ultra_sync:
                        new_fps = fps * segment.ultra_sync
                        # all_filters[i].append(Filter('minterpolate', fps=new_fps, mi_mode='mci'))
                        all_filters[i].append(Filter('framerate', fps=new_fps))
                    all_filters[i].extend([Filter('trim', start=offset), Filter('setpts', f'PTS-STARTPTS')])
                    if segment.ultra_sync:
                        # all_filters[i].append(Filter('minterpolate', fps=fps))
                        all_filters[i].append(Filter('framerate', fps=fps))

                if segment.fill_end:
                    if duration == durations[i]:
                        all_filters[i].append(Filter('split'))
                        all_outputs[i].append(f'filler_raw{suffix}')
                    else:
                        all_input_str[i].append(f'filler{suffix}')
                        all_filters[i][0].kw_params['n'] += 1

            v360 = Filter('v360', 'fisheye', 'e', 'lanc', iv_fov=segment.iv_fov, ih_fov=segment.ih_fov, h_fov=O_FOV, v_fov=O_FOV, alpha_mask=1, w=SIDE, h=SIDE)
            filters = [Filter(filter_str) for filter_str in segment.extra_video_filter] + [v360]

            filter_seqs.extend([
                FilterSeq(left_input_str, left_outputs, left_filters),
                FilterSeq(right_input_str, right_outputs, right_filters),
            ])
            if segment.fill_end:
                filter_seqs.insert(1, FilterSeq([f'filler_raw{suffix}'], [f'filler{suffix}'], [
                    Filter('trim', start=min(durations)),
                    Filter('setpts', f'PTS-STARTPTS'),
                ]))
                if duration == durations[1]:
                    filter_seqs.reverse()
            filter_seqs.extend([
                FilterSeq([f'left_raw{suffix}'], [f'left{suffix}'], filters),
                FilterSeq([f'right_raw{suffix}'], [f'right{suffix}'], filters),
                FilterSeq([f'left{suffix}', f'right{suffix}'], [f'overlay{suffix}'], [Filter('hstack', inputs=2)]),
                video_fs := FilterSeq([f'{input_index}:v:0', f'overlay{suffix}'], [f'video{suffix}'], [Filter('overlay')])
            ])

            fade_in = segment.fade[0]
            fade_out = segment.fade[1] if len(segment.fade) >= 2 else segment.fade[0]

            if fade_in:
                video_fs.filters.append(Filter('fade', t='in', st=0, d=fade_in))
            if fade_out:
                video_fs.filters.append(Filter('fade', t='out', st=duration - fade_out, d=fade_out))
            video_fs.filters.append(Filter('trim', duration=duration))

            if segment.do_audio:
                audio_inputs = [f'{i}:a:0' for i in (audio_inputs if segment.external_audio else [left_inputs, right_inputs][segment.channel])]
                audio_filters = [Filter(filter_str) for filter_str in segment.extra_audio_filter] + [Filter('concat', n=len(audio_inputs), v=0, a=1)]
                offset = segment.trim
                if cut_index == segment.channel:
                    offset += time_diff
                if offset > 0.0:
                    audio_filters.extend([Filter('atrim', start=offset), Filter('asetpts', f'PTS-STARTPTS')])

                if fade_in:
                    audio_filters.append(Filter('afade', t='in', st=0, d=fade_in))
                if fade_out:
                    audio_filters.append(Filter('afade', t='out', st=duration - fade_out, d=fade_out))
                audio_filters.append(Filter('atrim', duration=duration))
                filter_seqs.append(FilterSeq(audio_inputs, [f'audio{suffix}'], audio_filters))

            elif any_do_audio:
                ffmpeg_command.inputs.append(['-f', 'lavfi',  '-i', Filter(
                    'anullsrc', channel_layout=channel_layout, sample_rate=sample_rate).render()])
                filter_seqs.append(FilterSeq([f'{k}:a:0'], [f'audio{suffix}'], [Filter('atrim', duration=duration)]))
                k += 1

            input_index = k

            if segment_index > 1 and segment_index == len(segments):
                concat_inputs = []
                for i in range(1, segment_index + 1):
                    concat_inputs.extend([f'video{i}'])
                    if any_do_audio:
                        concat_inputs.extend([f'audio{i}'])
                if any_do_audio:
                    filter_seqs.append(FilterSeq(concat_inputs, ['video', 'audio'], [Filter('concat', n=len(segments), v=1, a=1)]))
                else:
                    filter_seqs.append(FilterSeq(concat_inputs, ['video'], [Filter('concat', n=len(segments), v=1, a=0)]))

    ffmpeg_command.filter_graph = FilterGraph(filter_seqs)
    if cfg.do_stab:
        ffmpeg_command.outputs.append(FFMpegCommand.Output(outputs=['-f', 'null', '-']))
    else:
        ffmpeg_command.outputs.append(video_output := FFMpegCommand.Output())
        video_output.mappings.extend(['-map', '[video]'])
        if any_do_audio and not cfg.separate_audio:
            video_output.mappings.extend(['-map', '[audio]'])
        video_output.codecs.extend(['-c:v', cfg.video_codec])
        if cfg.bitrate:
            video_output.codecs.extend(['-vb', cfg.bitrate])
        video_output.codecs.extend(['-pix_fmt', cfg.pixel_format])
        video_output.codecs.extend(['-preset', cfg.preset])

        if any_do_audio and not cfg.separate_audio:
            video_output.codecs.extend(['-c:a', cfg.audio_codec, '-ab', cfg.audio_bitrate])
        video_output.outputs.extend(['-t', fts(total_duration)])
        video_output.outputs.extend([out])

        if any_do_audio and cfg.separate_audio:
            ffmpeg_command.outputs.append(audio_output := FFMpegCommand.Output())
            audio_output.mappings.extend(['-map', f'[audio]'])
            audio_output.codecs.extend(['-c:a', 'pcm_s16le'])
            audio_output.outputs.extend(['-t', fts(total_duration)])
            audio_output.outputs.extend([out_wav])

    # import shlex
    # print(shlex.join(ffmpeg_command.as_list()))
    # exit()

    num_frames = round(float(total_duration * fps))
    if len(segments) > 1:
        print('=' * terminal_width)
        print(f'total_duration={d_to_hms(total_duration)} ({fts(total_duration)} s)')
    print(f'num_frames={num_frames}')
    print(f'Output Filename: "{out}"')

    if cfg.do_print:
        ffmpeg_command.print()
        exit()

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
            proc = pexpect.popen_spawn.PopenSpawn(ffmpeg_command.as_list())
        else:
            proc = pexpect.spawn(ffmpeg_command.as_list())
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
                    mul = eta = '?'
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
