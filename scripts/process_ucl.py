import pandas as pd
import numpy as np
import sys

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

ucl_to_devarda = {
    j: i for i, j in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 196, 197, 198, 200, 201, 208, 209, 210, 220, 225, 226, 227, 229, 234, 238, 240, 245])
}

devarda_total = 80

def read_indices(fn):
    my_dict = {}
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines[1:]:
            word, sentid, sentpos = line.strip().split(" ")
            my_dict[(int(sentid), int(sentpos))] = word
    return my_dict


def main():
    indices = read_indices(sys.argv[1])
    
    # read de Varda dataset
    df = pd.read_csv(sys.argv[2])
    # information about sentid and sentpos
    try:
        assert all([colname in df.columns for colname in ["total_responses","word_responses","clozeprob_0","clozesurp_0"]])
    except AssertionError:
        raise AssertionError("Current De Varda CSV doesn't contain data from cloze responses, run process_devarda_responses.py first")
    df_keys = df[["sent_id", "context_length"]].drop_duplicates()
    df_keys = set(zip(df_keys["sent_id"]-1, df["context_length"]+1))

    add_k = float(sys.argv[3])
    possible_V = int(sys.argv[4])
    out = []
    for key in indices:
        word = indices[key]
        i, j = key
        # if the UCL sentence is in de Varda
        row = {'word': word, 'sentid' : i, 'sentpos' : j}
        base_prob = add_k / (devarda_total + (add_k * possible_V))
        if i in ucl_to_devarda:
            # cloze data wasn't collected for the first word;
            # need to apply zero-cloze rule here and change 'undefined' accordingly
            if j == 1:
                row['total_responses'] = 0 # we should separate the first word of sentences in De Varda from UCL sentences that aren't there. 
                if add_k == 0:
                    # No cloze data was collected here
                    row['clozeprob'] = "NaN"
                    row['clozesurp'] = "NaN"
                else:
                    row['clozeprob'] = base_prob
                    row['clozesurp'] = -np.log2(base_prob)
            elif (ucl_to_devarda[i], j) in df_keys:
                curr_df = df[(df["sent_id"] == ucl_to_devarda[i]+1) & (df["context_length"] == j-1)]
                assert word == curr_df["word"].item()
                count, word = curr_df['total_responses'].item(), curr_df['word_responses'].item()
                smoothed_cloze = (word + add_k) / (count + (add_k * possible_V))
                row["clozeprob"] = smoothed_cloze
                row["clozesurp"] = -np.log2(smoothed_cloze)
                row['total_responses'] = curr_df['total_responses'].item()
            else:
                raise ValueError(f"data for sentence {i}, word {j} not found in de Varda data")
        else:
            row["clozeprob"] = "NaN"
            row['clozesurp'] = "NaN" # for UCL rows that were not in De Varda
        out.append(row)
    
    # using the same format as provo
    pd.DataFrame(out).to_csv(sys.stdout, index = False, sep=' ', na_rep='NaN')

if __name__ == "__main__":
    main()
