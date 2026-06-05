import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['CUDA_VISIBLE_DEVICES']= "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from argparse import ArgumentParser
from datasets import load_dataset

import sys
sys.path.append("../common")
from prompts import SYSTEM_PROMPT


def preprocess_function(example):
    return {"messages":[{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"BENGALI POEM: {example['bengali_version']}"},
                        {"role": "assistant", "content": f"<translation>\n{example['english_version']}\n</translation>"}]}


def main(model_name: str, gradient_acc_steps: int, lr: float):
    print('Loading dataset...')
    dataset = load_dataset('csv', data_files="../../data/SFT_data/train.csv", split='train')

    print('Formatting dataset...')
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    #tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)

    # Have to use a reworked template that supports assistant loss masking.
    #with open("../common/all_assistant.jinja", "r") as file:
    #    tokenizer.chat_template = file.read()

    training_args = SFTConfig(
        output_dir="./bn_en_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_acc_steps,
        learning_rate=lr,
        logging_steps=5,
        assistant_only_loss=False, # Setting this to true now - otherwise it was degrading outputs.
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
    #    processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Formatted sample...{trainer.processing_class.decode(trainer.train_dataset[0]['input_ids'])}")

    print("Starting training...")
    trainer.train()

    print('Training done, saving model...')
    trainer.save_model()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--gradient_acc_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()
    main(args.model_name, args.gradient_acc_steps, args.learning_rate)
