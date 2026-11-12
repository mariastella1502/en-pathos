# This script runs the discovery model on a batch of videos, extracting frames and audio, transcribing the audio, and generating observations based on a taxonomy of empathy gaps. The results are saved to JSON files for further analysis. It also generates a report summarizing the processing status of each video.

import argparse
import csv
import json
import random
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_discovery as core


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def select_videos(folder, count, seed, manifest, reset):
    if manifest.exists() and not reset:
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        videos = [Path(p) for p in previous['videos']]
        missing = [str(p) for p in videos if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'Manifest references missing videos: {missing[:5]}')
        return videos
    candidates = sorted(p.resolve() for p in folder.rglob('*.mp4') if p.is_file())
    if len(candidates) < count:
        raise RuntimeError(f'Found {len(candidates)} MP4 files under {folder}; need {count}. Prepare more scenes first.')
    videos = sorted(random.Random(seed).sample(candidates, count))
    save_json(manifest, {'seed': seed, 'count': count, 'videos': [str(p) for p in videos]})
    return videos


def normalize_english(analysis, mode, taxonomy, frames, segments, screenshots):
    # The V2 validator expects Italian metadata names. Provide compatible aliases
    # without modifying the translated source taxonomy.
    adapted = json.loads(json.dumps(taxonomy))
    for gap in adapted['gaps'].values():
        gap['label_it'] = gap.get('label_en', gap.get('label_it', ''))
        for behavior in gap['behaviors'].values():
            behavior['definition_it'] = behavior.get('definition_en', behavior.get('definition_it', ''))
    result = core.normalize_analysis(analysis, mode, adapted, frames, segments, screenshots)
    for observation in result['observations']:
        if 'behavior_definition_it' in observation:
            observation['behavior_definition_en'] = observation.pop('behavior_definition_it')
        if 'gap_label_it' in observation:
            observation['gap_label_en'] = observation.pop('gap_label_it')
    return result


def process_video(video, args, model, processor, config, taxonomy):
    out = args.out
    modes = ['guided', 'open'] if args.mode == 'both' else [args.mode]
    targets = {mode: out / f'{video.stem}_{mode}.json' for mode in modes}
    # Avoid repeated Whisper transcription for the same video on reruns.
    transcript_file = out / f'{video.stem}_transcript.json'
    if all(t.exists() and complete(t) for t in targets.values()):
        print(f'SKIP completed video: {video.name}', flush=True)
        return
    with tempfile.TemporaryDirectory() as temporary:
        audio = Path(temporary) / 'audio.wav'
        core.extract_audio(video, audio)
        if transcript_file.exists():
            transcription = json.loads(transcript_file.read_text(encoding='utf-8'))
        else:
            transcription = core.mlx_whisper.transcribe(
                str(audio), path_or_hf_repo='mlx-community/whisper-large-v3-turbo', word_timestamps=False
            )
            save_json(transcript_file, transcription)
        total = core.duration(video)
        starts = np.arange(0, total, args.stride)
        for mode in modes:
            target = targets[mode]
            if target.exists():
                document = json.loads(target.read_text(encoding='utf-8'))
                if document.get('video') != str(video):
                    raise RuntimeError(f'Output collision: {target} belongs to a different video')
            else:
                document = {'video': str(video), 'mode': mode, 'model': args.model,
                            'taxonomy_version': taxonomy.get('version') if mode == 'guided' else None,
                            'windows': []}
            done = {w['window_id'] for w in document['windows']}
            for index, start in enumerate(starts):
                if index in done:
                    continue
                end = min(total, float(start) + args.window)
                frame_dir = out / 'frames' / video.stem / f'window_{index:03d}'
                frame_dir.mkdir(parents=True, exist_ok=True)
                frames = core.extract_frames(video, float(start), end, frame_dir)
                screenshots = {t: str(Path(path).resolve()) for t, path in frames}
                segments = [{'start': s['start'], 'end': s['end'], 'text': s['text']}
                            for s in transcription['segments'] if s['end'] >= start and s['start'] <= end]
                acoustics = core.acoustic_features(audio, float(start), end)
                prompt = core.build_prompt(mode, segments, acoustics, frames, taxonomy)
                paths = [path for _, path in frames]
                formatted = core.apply_chat_template(processor, config, prompt, num_images=len(paths))
                output = core.generate(model, processor, formatted, paths,
                                       max_tokens=args.max_tokens, temperature=0.0, verbose=False)
                raw = output.text if hasattr(output, 'text') else str(output)
                analysis = normalize_english(core.parse_json(raw), mode, taxonomy, frames, segments, screenshots)
                document['windows'].append({
                    'window_id': index, 'start': round(float(start), 2), 'end': round(end, 2),
                    'frames': [{'timestamp_seconds': t, 'screenshot_path': screenshots[t]} for t, _ in frames],
                    'analysis': analysis,
                })
                document['windows'].sort(key=lambda w: w['window_id'])
                save_json(target, document)
                print(f'{video.name} | {mode} | window {index + 1}/{len(starts)}', flush=True)


def complete(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        windows = data['windows']
        return bool(windows) and all('analysis' in w for w in windows) and windows[-1]['end'] >= core.duration(Path(data['video'])) - 0.05
    except (OSError, ValueError, KeyError, TypeError):
        return False


def report(videos, args):
    rows = []
    for video in videos:
        for mode in (['guided', 'open'] if args.mode == 'both' else [args.mode]):
            path = args.out / f'{video.stem}_{mode}.json'
            row = {'video': video.name, 'mode': mode, 'status': 'missing', 'windows': 0,
                   'observations': 0, 'validation_errors': 0, 'distinct_behaviors': 0, 'gaps': ''}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                    observations = [o for w in data['windows'] for o in w['analysis'].get('observations', [])]
                    row.update(status='complete' if complete(path) else 'partial', windows=len(data['windows']),
                               observations=len(observations),
                               validation_errors=sum(len(w['analysis'].get('validation_errors', [])) for w in data['windows']),
                               distinct_behaviors=len({o['behavior'] for o in observations}),
                               gaps=';'.join(sorted({o['gap'] for o in observations})))
                except (OSError, ValueError, KeyError, TypeError):
                    row['status'] = 'invalid_json'
            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'batch_report.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Report: {args.out / "batch_report.csv"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', type=Path, default=Path('data/processed/meld/scenes'))
    parser.add_argument('--out', type=Path, default=Path('data/processed/meld/batch_40'))
    parser.add_argument('--taxonomy', type=Path, default=Path('data/benchmarks/empathy_taxonomy_detailed_en.json'))
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reset-selection', action='store_true')
    parser.add_argument('--mode', choices=['guided', 'open', 'both'], default='both')
    parser.add_argument('--model', default='mlx-community/Qwen2.5-VL-7B-Instruct-4bit')
    parser.add_argument('--window', type=float, default=12)
    parser.add_argument('--stride', type=float, default=9)
    parser.add_argument('--max-tokens', type=int, default=2200)
    args = parser.parse_args()
    if args.count < 1 or args.window <= 0 or not (0 < args.stride <= args.window):
        parser.error('Require count >= 1, window > 0, and 0 < stride <= window')
    if not args.taxonomy.is_file():
        parser.error(f'Missing taxonomy: {args.taxonomy}')
    taxonomy = json.loads(args.taxonomy.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    videos = select_videos(args.scenes, args.count, args.seed, args.out / 'batch_manifest.json', args.reset_selection)
    print(f'Selected {len(videos)} videos; loading model once...', flush=True)
    model, processor = core.load(args.model)
    config = core.load_config(args.model)
    errors = []
    for number, video in enumerate(videos, 1):
        print(f'[{number}/{len(videos)}] {video.name}', flush=True)
        try:
            process_video(video, args, model, processor, config, taxonomy)
        except Exception as exc:
            errors.append({'video': str(video), 'error': repr(exc), 'traceback': traceback.format_exc()})
            save_json(args.out / 'batch_errors.json', errors)
            print(f'ERROR {video.name}: {exc}; continuing', flush=True)
        report(videos, args)
    print(f'Finished. Videos with errors: {len(errors)}. Human validation required.')


if __name__ == '__main__':
    main()
