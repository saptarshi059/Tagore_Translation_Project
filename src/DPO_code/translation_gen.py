from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch


class TranslationGenDS(Dataset):
    def __init__(self, base_ds, tok):
        self.ds = base_ds
        self.tokenizer = tok

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        instance = self.ds[idx]
        tokenized_sample = self.tokenizer(instance, return_tensors='pt', padding=True)
        return tokenized_sample



def main():
    print('Loading SFT model...')
    checkpoint = '../SFT_code/bn_en_model/'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, fix_mistral_regex=True)

    base_poems = pd.read_csv('../../data/DPO_data/base_poems.csv').to_dict('records')
    torch_ds = TranslationGenDS(base_poems, tokenizer)
    torch_dataloader = DataLoader(torch_ds, batch_size=4, shuffle=False)

    for batch in torch_dataloader:
        print(batch)
        exit()


    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype="auto",
        device_map="auto"
    )





if __name__ == "__main__":
    main()