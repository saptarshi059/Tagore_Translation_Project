import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['CUDA_VISIBLE_DEVICES']= "0"

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import torch

import sys
sys.path.append("../common")
from prompts import LLM_JUDGE_PROMPT


class TranslationGenDS(Dataset):
    def __init__(self, base_ds, tok):
        self.ds = base_ds.to_dict('records')
        self.tokenizer = tok

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        instance = self.ds[idx]
        translation_string = ""
        for idx, translation in enumerate(instance['translations']):
            translation_string += f"TRANSLATION {idx}: {translation}\n"

        formatted_instance = [{"role": "system", "content": LLM_JUDGE_PROMPT},
                              {"role": "user", "content": f"BENGALI POEM: {instance['content']}\n {translation_string}"}]
        text = self.tokenizer.apply_chat_template(
            formatted_instance,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_sample = self.tokenizer(text)
        return tokenized_sample


def main():
    print('Loading Judge model...')

    model_name = "Qwen/Qwen3-32B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    parsed_df = pd.read_csv('../../data/DPO_data/parsed_translations.csv')
    torch_ds = TranslationGenDS(parsed_df, tokenizer)

    print(f"Sample formatted data: {torch_ds[0]}")

    torch_dataloader = DataLoader(torch_ds, batch_size=4, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))

    generations = []
    with torch.no_grad():
        for batch in tqdm(torch_dataloader):
            generated_ids = model.generate(
                input_ids = batch['input_ids'].to(model.device),
                attention_mask = batch['attention_mask'].to(model.device),
                max_new_tokens=100,
            )
            generations.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    print("Saving generations...")
    parsed_df['raw_generations'] = generations
    parsed_df.to_csv('../../data/DPO_data/raw_preferences.csv', index=False)

if __name__ == "__main__":
    main()