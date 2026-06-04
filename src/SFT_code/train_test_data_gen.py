import pandas as pd

parallel_poems = pd.read_csv("../../data/parsed_source/parallel_text.csv")

test_sample = 'Where the mind is without fear'
test = parallel_poems.query('english_title == @test_sample')
test.to_csv("../../data/SFT_data/test.csv", index=False)

parallel_poems = parallel_poems[parallel_poems['english_title'] != test_sample]
parallel_poems.to_csv("../../data/SFT_data/train.csv", index=False)