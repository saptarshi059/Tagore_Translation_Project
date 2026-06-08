from pathlib import Path
import pandas as pd
import re
import sys

sys.path.append("../../src/common")
from prompts import SYSTEM_PROMPT

if __name__ == "__main__":
    base_path = Path("../../data/DPO_data")

    raw_translations = pd.read_csv(base_path / 'raw_translations.csv').to_dict('records')

    parsable_translations = []
    for row in raw_translations:
        search_obj = re.search(r"<translation>(.*)</translation>", row["raw_generations"].split(SYSTEM_PROMPT)[1],
                               re.DOTALL)
        if search_obj:
            row['parsed_translation'] = search_obj[1].strip()
            parsable_translations.append(row)

    # Keeping those translations that have at least two clean generations
    parsed_df = pd.DataFrame(parsable_translations)
    parsed_df = parsed_df[parsed_df.groupby(['name', 'collection'])['name'].transform('count') >= 2]

    final_rows = []
    for _, group in parsed_df.groupby(['name', 'collection']):
        group_dict = group.to_dict('records')
        translations = []
        for row in group_dict:
            translations.append(row['parsed_translation'])
        final_rows.append({
            "name": group_dict[0]['name'],
            "collection": group_dict[0]['collection'],
            "content": group_dict[0]['content'],
            "translations": translations
        })

    pd.DataFrame(final_rows).to_parquet(base_path / "parsed_translations.parquet", index=False)