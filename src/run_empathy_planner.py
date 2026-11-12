# This script runs the empathy planner on a small set of scenarios and saves the results to a JSONL file.


import json
from pathlib import Path

from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen3-4B-4bit"

DATASET = Path("data/benchmarks/empathy_scenarios_en.jsonl")
PLANNER_PROMPT = Path("configs/planner_prompt.txt")
OUTPUT = Path("experiments/planner_responses_v2_en.jsonl")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

model, tokenizer = load(MODEL_ID)

scenarios = [
    json.loads(line)
    for line in DATASET.read_text(encoding="utf-8").splitlines()
    if line.strip()
][:3]

planner_instructions = PLANNER_PROMPT.read_text(
    encoding="utf-8"
)

generator_instructions = Path(
    "configs/response_generator_prompt.txt"
).read_text(encoding="utf-8")


def ask_model(system_prompt, user_prompt, max_tokens=300):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    ).strip()


with OUTPUT.open("w", encoding="utf-8") as f:
    for scenario in scenarios:
        print(f"Processing {scenario['id']}...")

        user_input = (
            f"Context:\n{scenario['context']}\n\n"
            f"Message:\n{scenario['user_message']}"
        )

        plan_text = ask_model(
            planner_instructions,
            user_input,
            max_tokens=350,
        )

        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            plan = {"raw_plan": plan_text}

        response = ask_model(
            generator_instructions,
            (
                f"{user_input}\n\n"
                "Empathic communication plan:\n"
                f"{json.dumps(plan, ensure_ascii=False)}\n\n"
                "Generate the final response."
            ),
            max_tokens=180,
        )

        result = {
            "id": scenario["id"],
            "primary_gap": scenario["primary_gap"],
            "plan": plan,
            "response": response,
            "model": MODEL_ID,
            "condition": "planner",
        }

        f.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )
        f.flush()

        print("Plan:", plan)
        print("Response:", response)
        print()

print(f"Saved to {OUTPUT}")
