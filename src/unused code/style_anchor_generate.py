from pathlib import Path
from tqdm import tqdm
import pandas as pd
import bm25s

def main():
    base_path = Path("../../data/parsed_source")

    print('Loading Shakespeare poems...')
    shakespeare_poems = pd.read_csv(base_path / "shakespeare_parsed.csv")

    print('Loading parallel poems...')
    parallel_poems = pd.read_csv(base_path / 'parallel_text.csv')

    corpus = shakespeare_poems.sonnet.to_list()

    print('Creating index...')
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)

    print('Locating best themed Shakespearean equivalent for Tagore translation...')
    best_match = []
    for row in tqdm(parallel_poems.itertuples()):
        eng_translation = row.english_version
        eng_translation_tokens = bm25s.tokenize(eng_translation)
        docs, _ = retriever.retrieve(eng_translation_tokens, k=1)
        best_match.append(str(docs[0, 0]))

    print('Saving final dataset...')
    parallel_poems['nearest_shakespeare_sonnet'] = best_match

    test_sample = 'Where the mind is without fear'
    test = parallel_poems.query('english_title == @test_sample')
    test.to_csv(base_path / "test.csv", index=False)

    parallel_poems = parallel_poems[parallel_poems['english_title'] != test_sample]
    parallel_poems.to_csv(base_path / "train.csv", index=False)


if __name__ == "__main__":
    main()