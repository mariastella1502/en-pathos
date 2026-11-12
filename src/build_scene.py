#This script takes a specific dialogue from the MELD dataset, extracts the corresponding video segments, normalizes them to a consistent format, and concatenates them into a single video file representing the entire scene. The output video is saved in the specified output directory.

import json
import subprocess
import tempfile
from pathlib import Path

DATA_DIR = Path("data/raw/meld/MELD.Raw")
JSONL = Path("data/processed/meld/train_dialogues.jsonl")
OUTPUT_DIR = Path("data/processed/meld/scenes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIALOGUE_ID = "train_4"

with JSONL.open(encoding="utf-8") as f:
    dialogue = next(
        json.loads(line)
        for line in f
        if json.loads(line)["id"] == DIALOGUE_ID
    )

video_paths = []

for utt in dialogue["utterances"]:
    matches = list(DATA_DIR.rglob(utt["video_filename"]))
    if not matches:
        raise FileNotFoundError(utt["video_filename"])
    video_paths.append(matches[0].resolve())

output = OUTPUT_DIR / f"{DIALOGUE_ID}.mp4"

with tempfile.TemporaryDirectory() as temp_dir:
    temp = Path(temp_dir)
    normalized = []

    for i, path in enumerate(video_paths):
        segment = temp / f"segment_{i:03d}.mp4"

        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(path),
            "-vf", "scale=640:360:force_original_aspect_ratio=decrease,"
                   "pad=640:360:(ow-iw)/2:(oh-ih)/2",
            "-r", "24",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            str(segment)
        ], check=True)

        normalized.append(segment)

    concat_file = temp / "concat.txt"

    with concat_file.open("w") as f:
        for path in normalized:
            f.write(f"file '{path}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output)
    ], check=True)

print(f"Scene created: {output}")
