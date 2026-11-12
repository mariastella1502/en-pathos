#This script takes 40 dialogue from the MELD dataset, extracts the corresponding video segments, normalizes them to a consistent format, and concatenates them into a single video file representing the entire scene. The output video is saved in the specified output directory.

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

DATA_DIR = Path("data/raw/meld/MELD.Raw")
JSONL = Path("data/processed/meld/train_dialogues.jsonl")
OUTPUT_DIR = Path("data/processed/meld/scenes")


def run_ffmpeg(command):
    subprocess.run(
        command,
        check=True
    )


def find_video(filename):
    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Video not found: {filename}"
        )

    return matches[0].resolve()


def build_scene(dialogue):
    dialogue_id = dialogue["id"]
    output = OUTPUT_DIR / f"{dialogue_id}.mp4"

    if output.exists():
        print(f"SKIP: {dialogue_id} already exists")
        return True

    video_paths = [
        find_video(utt["video_filename"])
        for utt in dialogue["utterances"]
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        normalized = []

        for i, path in enumerate(video_paths):
            segment = temp / f"segment_{i:03d}.mp4"

            run_ffmpeg([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-vf",
                "scale=640:360:force_original_aspect_ratio=decrease,"
                "pad=640:360:(ow-iw)/2:(oh-ih)/2",
                "-r", "24",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-ar", "44100",
                "-ac", "2",
                str(segment)
            ])

            normalized.append(segment)

        concat_file = temp / "concat.txt"

        with concat_file.open("w", encoding="utf-8") as f:
            for path in normalized:
                f.write(f"file '{path}'\n")

        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output)
        ])

    print(f"CREATED: {output}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be >= 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with JSONL.open(encoding="utf-8") as f:
        dialogues = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    # Select reproducibly from the training dialogues.
    dialogues = sorted(dialogues, key=lambda d: d["id"])

    if len(dialogues) < args.count:
        raise RuntimeError(
            f"Only {len(dialogues)} dialogues available; "
            f"requested {args.count}"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(dialogues, args.count)

    completed = 0
    failed = []

    for index, dialogue in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] "
            f"Building {dialogue['id']}",
            flush=True
        )

        try:
            build_scene(dialogue)
            completed += 1
        except Exception as exc:
            failed.append({
                "dialogue_id": dialogue["id"],
                "error": str(exc)
            })
            print(f"ERROR: {dialogue['id']}: {exc}")

    report = {
        "requested": args.count,
        "completed": completed,
        "failed": failed,
        "selected_dialogues": [
            d["id"] for d in selected
        ]
    }

    report_path = OUTPUT_DIR / "build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(
        f"\nFinished: {completed}/{args.count} "
        f"scenes available."
    )


if __name__ == "__main__":
    main()
