import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['CUDA_VISIBLE_DEVICES']= "0"

from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from argparse import ArgumentParser
from datasets import load_dataset
from prompts import SYSTEM_PROMPT


def preprocess_function(example):
    return {"messages":[{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"BENGALI POEM (TAGORE): {example['bengali_version']}\n\n"
                                                    f"RELATED ENGLISH POEM (SHAKESPEARE): {example['nearest_shakespeare_sonnet']}"},
                        {"role": "assistant", "content": f"TRANSLATION: {example['english_version']}"}]}


def main(model_name):
    print('Loading dataset...')
    dataset = load_dataset('csv', data_files="../data/parsed_source/train.csv", split='train')

    print('Formatting dataset...')
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    training_args = SFTConfig(
        output_dir="./bn_en_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=4e-5,
        logging_steps=5,
        assistant_only_loss=True,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()

    print('Training done, saving model...')
    trainer.save_model()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-1.7B-Base")
    args = parser.parse_args()
    main(args.model_name)
