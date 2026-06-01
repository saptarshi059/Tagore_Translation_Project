import os

os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from datasets import load_dataset
import torch


def preprocess_function(example):
    return {"messages": [{"role": "system", "content": "Translate the given Bengali poem to English, in the style of William Shakespeare."},
                         {"role": "user", "content": f"BENGALI POEM (TAGORE): {example['bengali_version']}"
                          }]
            }


def main(model_name):
    print('Loading dataset...')
    dataset = load_dataset('csv', data_files="../data/parsed_source/test.csv", split='train')

    gold = dataset[0]['english_version']
    print(f'Original Work (Gold): {gold}\n')

    print('Formatting dataset...')
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)

    text = tokenizer.apply_chat_template(
        dataset['messages'][0],  # Since there is only 1 test sample
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print('Generating...')
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=300)

    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Replicated Style (translation): {decoded_text}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="bn_en_model")
    args = parser.parse_args()
    main(args.model_name)
