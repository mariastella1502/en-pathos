#This script analyzes a specific scene from the MELD dataset using the Qwen2.5-VL model. It extracts frames from the video, processes the dialogue transcript, and generates an analysis based on the provided instructions.

import json
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

SCENE = Path("data/processed/meld/scenes/train_4.mp4")
DIALOGUES = Path("data/processed/meld/train_dialogues.jsonl")
PROMPT = Path("configs/empathy_prompt.txt")
OUTPUT = Path("experiments/train_4_analysis.txt")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
NUM_FRAMES = 8

# Load dialogue and transcript.
with DIALOGUES.open(encoding="utf-8") as f:
    dialogue = next(
        item for line in f
        if (item := json.loads(line))["id"] == "train_4"
    )

transcript = "\n".join(
    f'{utt["speaker"]}: {utt["text"]}'
    for utt in dialogue["utterances"]
)

instructions = PROMPT.read_text(encoding="utf-8")

# Determine scene duration.
probe = subprocess.run(
    [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(SCENE),
    ],
    capture_output=True,
    text=True,
    check=True,
)
duration = float(probe.stdout.strip())

# Extract frames at evenly spaced points.
frames = []

with tempfile.TemporaryDirectory() as tmp:
    for i in range(NUM_FRAMES):
        timestamp = duration * (i + 0.5) / NUM_FRAMES
        frame_path = Path(tmp) / f"frame_{i}.jpg"

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(timestamp),
                "-i", str(SCENE),
                "-frames:v", "1",
                "-vf", "scale=448:-2",
                str(frame_path),
            ],
            check=True,
        )
        frames.append(Image.open(frame_path).convert("RGB").copy())

print(f"Extracted {len(frames)} frames.")

# Load model.
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to("mps")
model.eval()

# Construct multimodal message.
content = [
    {"type": "text", "text": (
        instructions
        + "\n\nDialogue transcript:\n"
        + transcript
        + "\n\nAnalyse the following video frames in order."
    )}
]

for frame in frames:
    content.append({"type": "image", "image": frame})

messages = [
    {"role": "user", "content": content}
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=frames,
    padding=True,
    return_tensors="pt",
)

inputs = {
    key: value.to("mps") if isinstance(value, torch.Tensor)
    else value
    for key, value in inputs.items()
}

print("Analysing scene...")

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=650,
        do_sample=False,
    )

generated = generated[:, inputs["input_ids"].shape[1]:]

response = processor.batch_decode(
    generated,
    skip_special_tokens=True,
)[0]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(response, encoding="utf-8")

print("\n=== EN PATHOS ANALYSIS ===\n")
print(response)
print(f"\nSaved to {OUTPUT}")
