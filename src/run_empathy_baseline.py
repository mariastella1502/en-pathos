# Run empathy baseline experiment


import json
from pathlib import Path

from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen3-4B-4bit"

DATASET_PATH = Path("data/benchmarks/empathy_scenarios_en.jsonl")
OUTPUT_PATH = Path("experiments/baseline_responses.jsonl")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading model...")
model, tokenizer = load(MODEL_ID)

with DATASET_PATH.open(encoding="utf-8") as f:
    scenarios = [json.loads(line) for line in f if line.strip()][:3]

print(f"Loaded {len(scenarios)} scenarios.")

SYSTEM_PROMPT = (
    "You are a helpful conversational assistant. "
    "Respond naturally to the user."
)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for i, scenario in enumerate(scenarios, start=1):
        print(f"\n[{i}/{len(scenarios)}] {scenario['id']}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{scenario['context']}\n\n"
                    f"Message:\n{scenario['user_message']}"
                ),
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False,
        )

        result = {
            "id": scenario["id"],
            "primary_gap": scenario["primary_gap"],
            "context": scenario["context"],
            "user_message": scenario["user_message"],
            "response": response.strip(),
            "model": MODEL_ID,
            "condition": "baseline",
        }

        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

        print("Response:", response[:200].replace("\n", " "))

print(f"\nSaved to {OUTPUT_PATH}")
