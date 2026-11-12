#This script takes the MELD dataset and converts it into a JSONL format that is compatible with the en-pathos project. It reads the original CSV files, processes the dialogues and utterances, and saves them in a structured format for further analysis.

# Paths relative to the en-pathos project root
DATA_DIR = Path("data/raw/meld/MELD.Raw")
OUTPUT_DIR = Path("data/processed/meld")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

for split, filename in files.items():
    csv_path = DATA_DIR / filename
    df = pd.read_csv(csv_path)

    dialogues = []

    for dialogue_id, group in df.groupby("Dialogue_ID", sort=True):
        group = group.sort_values("Utterance_ID")
        first = group.iloc[0]

        utterances = []

        for _, row in group.iterrows():
            utterance_id = int(row["Utterance_ID"])

            utterances.append({
                "speaker": str(row["Speaker"]),
                "text": str(row["Utterance"]),
                "emotion": str(row["Emotion"]),
                "sentiment": str(row["Sentiment"]),
                "start_time": str(row["StartTime"]),
                "end_time": str(row["EndTime"]),
                "video_filename": (
                    f"dia{int(dialogue_id)}_utt{utterance_id}.mp4"
                ),
            })

        dialogues.append({
            "id": f"{split}_{int(dialogue_id)}",
            "source": "MELD",
            "split": split,
            "season": int(first["Season"]),
            "episode": int(first["Episode"]),
            "utterances": utterances,
        })

    output_path = OUTPUT_DIR / f"{split}_dialogues.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for dialogue in dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")

    print(f"{split}: {len(dialogues)} dialogues saved")

print("\nMELD preprocessing completed!")
