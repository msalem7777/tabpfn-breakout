"""Render a verified emulator replay with observable TabPFN diagnostics.

No model inference and no teacher calls. Targets come only from the saved run.
Replay stops on the first state/outcome mismatch; no invented game footage.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from breakout_pfn.config import ExperimentConfig
from breakout_pfn.runtime import environment
from breakout_pfn.student_controller import ControllerTrace
from breakout_pfn.experiment_inputs import read_state, load_bank
from breakout_pfn.runtime import brick_layout
from breakout_pfn.ball_paddle import fingerprint, outcome
from breakout_pfn.runtime import write_json

BG = '#0b1120'
WHITE = '#f2f5fb'
MUTED = '#9caec9'
CYAN = '#44e2db'
GOLD = '#ffc773'
PANEL = '#151f33'


@lru_cache(maxsize=16)
def font(size):
    for path in ('C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def compose(observation, row, x, y, history, total, speed, backend):
    """Selected examples and outputs are observable, not an attention map."""
    canvas = Image.new('RGB', (1080, 1080), BG)
    draw = ImageDraw.Draw(canvas)
    def text(xp, yp, value, size=23, fill=WHITE):
        draw.text((xp, yp), str(value), font=font(size), fill=fill)
    text(40, 25, 'TabPFN plays Breakout', 44)
    text(42, 85, 'Teacher examples in. Paddle destinations out.', 24, MUTED)
    game = Image.fromarray(np.asarray(observation, np.uint8)).resize((480, 630), Image.Resampling.NEAREST)
    canvas.paste(game, (40, 192))
    draw.rounded_rectangle((550, 190, 1040, 851), 20, fill=PANEL)
    text(577, 211, 'THE DECISION', 19, CYAN)
    text(577, 243, f'Move paddle to x = {row["target"]*159:.1f} px', 29)
    text(577, 291, f'Frame {row["frame"]+1:,} / {total:,}', 20, MUTED)
    for xp, title, value in [(577, 'NET SCORE', row['score']), (748, 'HITS', row['hits']), (890, 'BRICKS LEFT', row['bricks'])]:
        text(xp, 337, title, 16, MUTED)
        text(xp, 366, f'{value:g}', 35)
    ids = np.asarray(row['context_selection']['bank_indices'], int)
    # Ball-position example panel omitted from the video.
    text(577, 666, 'Teacher destinations in this context', 19, MUTED)
    draw.line((580, 714, 1008, 714), fill='#3b4b64', width=2)
    for label in y[ids]:
        p = 580+float(label)*428
        draw.line((p, 702, p, 725), fill=CYAN, width=1)
    p = 580+row['target']*428
    draw.polygon([(p, 696), (p-6, 685), (p+6, 685)], fill=GOLD)
    text(577, 742, 'Gold marker: TabPFN prediction', 18, GOLD)
    # Two separate scales avoid equating game points with brick count.
    for k, left, title, limit, color in [('score', 40, 'NET SCORE OVER GAME FRAMES', 450, CYAN),
                                         ('bricks', 555, 'BRICKS REMAINING', 108, GOLD)]:
        text(left, 888, title, 17, MUTED)
        draw.line((left, 990, left+480, 990), fill='#3b4b64')
        points=[(left+r['frame']/max(total-1,1)*480, 987-max(0,r[k])/limit*66) for r in history]
        if len(points)>1:
            draw.line(points, fill=color, width=3)
        text(left, 994, '0', 14, MUTED)
        text(left+425, 994, f'{total:,}', 14, MUTED)
    text(40, 1024, 'Prediction diagnostics, not internal attention. Model weights stay fixed.', 19, MUTED)
    return canvas


def render(args):
    run = Path(args.run_dir)
    cfg = ExperimentConfig.load(run/'config.json')
    summary = json.loads((run/'summary.json').read_text())
    experiment = json.loads((run/'experiment.json').read_text())
    include_bricks = bool(summary.get('includes_bricks', False))
    x, y, _, _, _, _ = load_bank(args.data, 1, include_bricks)
    if fingerprint(x, y) != summary['context_sha256']:
        raise ValueError('Replay needs the exact teacher bank used in this run')
    rows = [json.loads(line) for line in (run/'predictions.jsonl').open() if line.strip()]
    if not rows or len(rows) != summary['frames']:
        raise ValueError('Prediction log is incomplete')
    if any(r['frame'] != i for i,r in enumerate(rows)):
        raise ValueError('Prediction frames must be consecutive from zero')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Choose a new output filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem+'.partial.mp4')
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('Install ffmpeg and place it on PATH before rendering')
    if not np.isfinite(args.speed) or args.speed <= 0 or args.fps < 1:
        raise ValueError('Positive speed and FPS required')
    # 60 emulator frames = one second of game time; repeated frames allow slower playback.
    stride = 60*args.speed/args.fps
    errorlog = output.with_suffix('.ffmpeg.log')
    controller_name = experiment['student_controller']
    initial = summary['initialization']
    score = hits = written = 0
    history = []
    next_frame = 0.0
    with errorlog.open('w') as stderr, environment(cfg) as env:
        controller = ControllerTrace(env, controller_name) if controller_name == 'analog-target' else None
        env.reset(seed=initial['seed'], start_paddle_x=initial['start_paddle_x'])
        if controller is None:
            controller = ControllerTrace(env, controller_name)
        process = subprocess.Popen([executable, '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '1080x1080', '-r', str(args.fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE, stderr=stderr)
        try:
            for i, row in enumerate(rows):
                state = read_state(env, include_bricks)
                if not np.array_equal(state, np.asarray(row['state'], np.float32)):
                    raise RuntimeError(f'Replay state diverged at frame {i}; no video certified')
                result = env.step_to_target(row['target'])
                score += result.reward
                hits += int(result.paddle_hit)
                bricks = int(brick_layout(result.observation).sum())
                if (score != row['score'] or hits != row['hits'] or bricks != row['bricks']
                        or outcome(result, bricks) != row.get('event')):
                    raise RuntimeError(f'Replay outcome diverged at frame {i}')
                if controller.last != row['controller']:
                    raise RuntimeError(f'Replay controller diverged at frame {i}')
                history.append(row)
                if i >= next_frame or i == len(rows)-1:
                    panel = compose(result.observation, row, x, y, history, len(rows), args.speed,
                                    summary.get('backend', summary['model_settings']['backend']))
                    while i >= next_frame or (i == len(rows)-1 and next_frame < len(rows)):
                        process.stdin.write(panel.tobytes())
                        written += 1
                        next_frame += stride
                if (i+1)%1000 == 0:
                    print(f'[replay] verified={i+1}/{len(rows)} video_frames={written}', flush=True)
            # Hold the actual ending long enough to read. Never hide the loss.
            ending = compose(result.observation, rows[-1], x, y, history, len(rows), args.speed,
                             summary.get('backend', summary['model_settings']['backend']))
            draw = ImageDraw.Draw(ending)
            draw.rectangle((35, 79, 1040, 116), fill=BG)
            message = f'{hits} paddle hits. {108-bricks}/108 bricks cleared. Ended: {summary["status"]}.'
            draw.text((42, 85), message, font=font(24), fill=GOLD)
            for _ in range(3*args.fps):
                process.stdin.write(ending.tobytes())
                written += 1
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'ffmpeg failed; inspect {errorlog}')
        except BaseException:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    errorlog.unlink(missing_ok=True)
    write_json(output.with_suffix('.replay.json'), dict(verified_frames=len(rows),
        state_comparison='exact float32', controller_comparison='exact logged action',
        video_frames=written, speed=args.speed, fps=args.fps, source=str(run),
        inference_during_replay=False, context_sha256=fingerprint(x,y)))
    print(f'[replay] saved={output} all {len(rows)} frames verified; no model calls')


def smoke_test():
    row = dict(frame=0,target=.5,score=0,hits=0,bricks=108,state=[.5,.4,0,.01,.5,0],
               context_selection={'bank_indices':[0,1]})
    canvas=compose(np.zeros((210,160,3),np.uint8),row,np.zeros((2,6)),np.array([.3,.7]),[row],10,4,'local')
    assert canvas.size==(1080,1080)
    print('Decision video smoke test passed (no emulator/model loaded)')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--run-dir')
    parser.add_argument('--data')
    parser.add_argument('--output')
    parser.add_argument('--speed',type=float,default=4.)
    parser.add_argument('--fps',type=int,default=30)
    args=parser.parse_args(argv)
    smoke_test()
    if args.self_test:return 0
    if not all((args.run_dir,args.data,args.output)):
        parser.error('Provide --run-dir, --data, and --output')
    render(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
