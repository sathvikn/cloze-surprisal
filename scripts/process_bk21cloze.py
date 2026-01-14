import pandas as pd
import numpy as np
import sys

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

def read_indices(fn):
    my_dict = {}
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines[1:]:
            word, sentid, sentpos = line.strip().split(" ")
            my_dict[(int(sentid), int(sentpos))] = word
    return my_dict

# ITEM indices start at 5 for some reason
def main():
    index_path = sys.argv[1]
    data_path = sys.argv[2]
    add_k = float(sys.argv[3])
    possible_V = int(sys.argv[4])
    bk_est_total = 90 # BK21 says that average, 90 participants responded to each stimulus 
    indices = read_indices(index_path)
    # read Brothers and Kuperberg dataset
    df = pd.read_csv(data_path)
    condition_to_idx = {"HC": 0, "MC": 1, "LC": 2}
    df["condition"] = df["condition"].map(condition_to_idx)
    df["ITEM"] -= 5
    df["sentid"] = 3 * df["ITEM"] + df["condition"]

    df_keys = df[["sentid", "position"]].drop_duplicates()
    df_keys = set(zip(df_keys["sentid"], df_keys["position"]))

    # sys.argv[3] is the column you'd like to print
    #print(f"word {sys.argv[3]}")
    cloze = []
    for key in indices:
        word = indices[key]
        i, j = key
        row = {"word" : word, "sentid": i, "sentpos": j, "clozeprob" : "inf", "clozesurp" : "inf"}
        if key in df_keys:
            # if the word is attested, modify the row.
            curr_df = df[(df["sentid"] == i) & (df["position"] == j)].iloc[0]
            assert word == curr_df["critical_word"], f"{word} {curr_df['critical_word']}"
            est_responses = np.ceil(curr_df['cloze'] * bk_est_total)
            clozeprob = (est_responses + add_k) / (bk_est_total + (add_k * possible_V))
            row['clozeprob'] = clozeprob
            row['clozesurp'] = -np.log2(clozeprob)
        # for all other non-critical words
        cloze.append(row)
    pd.DataFrame(cloze).to_csv(sys.stdout, index = False, sep=' ', na_rep='NaN')


if __name__ == "__main__":
    main()
