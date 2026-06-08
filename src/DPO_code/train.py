import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['CUDA_VISIBLE_DEVICES']= "0"

from transformers import AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
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

    print('Processing dataset...')
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

    print('Loading SFT checkpoint...')
    model = AutoModelForCausalLM.from_pretrained('../SFT_code/bn_en_model/', device_map='auto')

    training_args = DPOConfig(
        output_dir="./bn_en_model_DPO",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=5,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()

    print('Training done, saving model...')
    trainer.save_model()


if __name__ == "__main__":
    main()