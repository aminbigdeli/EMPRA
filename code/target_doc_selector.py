
from tqdm import tqdm
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run', type=str, required=True, help="Path to re-ranked run by the black-box model")
    parser.add_argument('-output', type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    # Read the re-ranked run file
    reranked_run = pd.read_csv(args.run, sep="\t", names=['qid', 'pid', 'score'])

    # Group by 'qid'
    grps = reranked_run.groupby(['qid'])
    easy_output = []
    hard_output = []

    for name, group in tqdm(grps):
        group = group.reset_index(drop=True)
        group['rank'] = group.index + 1
        group = group[['qid', 'pid', 'rank', 'score']]
        easy_docs = group.iloc[50:100]
        if not easy_docs.empty:
            for i in range(0, 50, 10):
                easy_sample = easy_docs.iloc[i:i+10].sample(1)
                easy_output.append(easy_sample.values.tolist()[0])
        hard_docs = group.tail(5)
        if not hard_docs.empty:
            hard_output.extend(hard_docs.values.tolist())
            
    easy_df = pd.DataFrame(easy_output, columns=['qid', 'pid', 'rank', 'score'])
    easy_df['qid'] = easy_df['qid'].astype(int)
    easy_df['pid'] = easy_df['pid'].astype(int)
    easy_df['rank'] = easy_df['rank'].astype(int)
    easy_df = easy_df[['qid', 'pid', 'rank', 'score']]

    hard_df = pd.DataFrame(hard_output, columns=['qid', 'pid', 'rank', 'score'])
    hard_df['qid'] = hard_df['qid'].astype(int)
    hard_df['pid'] = hard_df['pid'].astype(int)
    hard_df['rank'] = hard_df['rank'].astype(int)
    hard_df = hard_df[['qid', 'pid', 'rank', 'score']]

    # Save the DataFrames to the specified output path
    easy_output_path = f"{args.output}/Easy-5.tsv"
    hard_output_path = f"{args.output}/Hard-5.tsv"
    easy_df.to_csv(easy_output_path, sep="\t", index=False, header=None)
    hard_df.to_csv(hard_output_path, sep="\t", index=False, header=None)

if __name__ == "__main__":
    main()
