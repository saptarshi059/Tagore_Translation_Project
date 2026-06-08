from pathlib import Path
import pandas as pd
import re
import sys
sys.path.append("../../src/common/")
from prompts import LLM_JUDGE_PROMPT


def main():
    base_path = Path("../../data/DPO_data")

    raw_preferences = pd.read_parquet(base_path / "raw_preferences.parquet")
    preference_data = []
    for row in raw_preferences.itertuples():
        search_str = row.raw_generations.split(LLM_JUDGE_PROMPT)[1]
        best = re.search(r"BEST TRANSLATION: (\d)", search_str, re.DOTALL)
        worst = re.search(r"WORST TRANSLATION: (\d)", search_str, re.DOTALL)

        if best and worst:
            preference_data.append({
                "poem": row.content,
                "best": row.translations[int(best[1])],
                "worst": row.translations[int(worst[1])],
            })

    pd.DataFrame(preference_data).to_parquet(base_path / "parsed_preferences.parquet", index=False)

if __name__ == "__main__":
    main()