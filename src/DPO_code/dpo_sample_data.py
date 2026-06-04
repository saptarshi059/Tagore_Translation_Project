import pandas as pd

all_poems = pd.read_csv("../../data/raw_source/poem.csv")
sampled_data = all_poems.query("collection != 'গীতাঞ্জলি'").sample(n=100, random_state=42)

sampled_data.to_csv('../../data/DPO_data/base_poems.csv', index=False)