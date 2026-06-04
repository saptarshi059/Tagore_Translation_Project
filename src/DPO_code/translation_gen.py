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
from prompts import SYSTEM_PROMPT


class TranslationGenDS(Dataset):
    def __init__(self, base_ds, tok):
        self.ds = base_ds.to_dict('records')
        self.tokenizer = tok

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        instance = self.ds[idx]['content']
        formatted_instance = [{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": f"ORIGINAL BENGALI: {instance}"}]
        text = self.tokenizer.apply_chat_template(
            formatted_instance,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_sample = self.tokenizer(text)
        return tokenized_sample


def main():
    print('Loading SFT model...')
    checkpoint = '../SFT_code/bn_en_model/'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, fix_mistral_regex=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype="auto",
        device_map="auto"
    )

    base_poems = pd.read_csv('../../data/DPO_data/base_poems.csv')
    torch_ds = TranslationGenDS(base_poems, tokenizer)
    torch_dataloader = DataLoader(torch_ds, batch_size=4, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))

    generations = []
    with torch.no_grad():
        for batch in tqdm(torch_dataloader):
            generated_ids = model.generate(
                input_ids = batch['input_ids'].to(model.device),
                attention_mask = batch['attention_mask'].to(model.device),
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=3
            )
            generations.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    print("Saving generations...")
    base_poems = base_poems.loc[base_poems.index.repeat(3)]
    base_poems['raw_generations'] = generations
    base_poems.to_csv('../../data/DPO_data/raw_translations.csv', index=False)

if __name__ == "__main__":
    main()