from trl import DPOTrainer
from datasets import load_dataset
import sys
sys.path.append("../../src/common")
from prompts import SYSTEM_PROMPT


def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": f"{SYSTEM_PROMPT}\nBENGALI POEM: {example['poem']}"}],
        "chosen": [{"role": "assistant", "content": f"<translation>\n{example['best']}\n</translation>"}],
        "rejected": [{"role": "assistant", "content": f"<translation>\n{example['worst']}\n</translation>"}],
    }


def main():
    dataset = load_dataset("parquet", data_files="../../data/DPO_data/parsed_preferences.parquet", split='train')
    dataset = dataset.map(preprocess_function, remove_columns=dataset.columns)
    print(next(iter(dataset)))

if __name__ == "__main__":
    main()