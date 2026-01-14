import pickle, os, re
import numpy as np
import pandas as pd
from process_ucl import read_indices, ucl_to_devarda
from process_devarda_responses import trgt_compregension, clean_cloze_questions, clean_cloze_responses, punct_words

devarda_folder = "cloze_data/devarda"
output_dir = "cloze_data/context_responses"

def get_all_responses(candidates):
    responses = []
    for candidate in candidates:
        candidate = str(candidate).strip()
        candidate = re.sub("[\.\,\?\!\:\;]", "", candidate)
        if candidate in punct_words.keys():
            candidate = punct_words[candidate]
        responses.append(candidate)
    return {
        "responses" : responses
    }

def load_clean_responses(listnum, items):
    cloze = pd.read_excel(f"{devarda_folder}/cloze_responses/list"+str(listnum)+".xls")
    part_comprehension = cloze.iloc[1:, 17:28]
    correctness = part_comprehension.iloc[:, 1:] == trgt_compregension
    part_comprehension["score"] = correctness.sum(axis=1)
    to_exclude = part_comprehension[part_comprehension.score < 8].iloc[:, 0]
    cloze = cloze[~cloze.index.isin(to_exclude.index)]
    qs_cloze = clean_cloze_questions(cloze.iloc[0,28:])
    print("Check =", qs_cloze[~(qs_cloze.values == items['sentence'].values)].empty)
    resp_cloze = cloze.iloc[1:, 28:]
    spelled, _ = clean_cloze_responses(resp_cloze)
    spell_transposed = spelled.transpose()
    cloze_results = []
    for n in range(len(spell_transposed)):
        candidates = list(spell_transposed.iloc[n,:])
        cloze_results.append(get_all_responses(candidates))
    return pd.DataFrame(cloze_results)

if __name__ == "__main__":
    df = pd.read_csv(f"{devarda_folder}/all_measures.csv")
    sentences = df.apply(lambda row : f"{row['item']} {row['word']}", axis = 1)

    item_set = pd.read_csv(f"{devarda_folder}/item-set.csv", dtype={ 'sentence':np.str_, 'word':np.str_, 'word2':np.str_})
    item_set["sentence"] = item_set["sentence"].str.strip()

    items = {i : item_set[item_set["list"] == i] for i in range(1, 9)}
    cloze = pd.concat([load_clean_responses(i, items[i]) for i in range(1, 9)], ignore_index=True) # combines the cloze responses in each list
    df = pd.concat([df, cloze], axis = 1) # merges the unsmoothed cloze results/response details with the original data
    
    # now combine with UCL indices (copying from process_ucl.py)
    indices = read_indices("text_data/ucl.indices")
    df_keys = df[["sent_id", "context_length"]].drop_duplicates()
    df_keys = set(zip(df_keys["sent_id"]-1, df["context_length"]+1))
    out = {}
    for key in indices:
        word = indices[key]
        i, j = key
        # if the UCL sentence is in de Varda
        if i in ucl_to_devarda:
            # cloze data wasn't collected for the first word;
            # need to apply zero-cloze rule here and change 'undefined' accordingly
            if j == 1:
                out[(i, j)] = []
            elif (ucl_to_devarda[i], j) in df_keys:
                curr_df = df[(df["sent_id"] == ucl_to_devarda[i]+1) & (df["context_length"] == j-1)]
                assert word == curr_df["word"].item()
                out[(i, j)] = curr_df['responses'].item()
            else:
                raise ValueError(f"data for sentence {i}, word {j} not found in de Varda data")

    with open(os.path.join(output_dir, "ucl_devarda_responses.pickle"), "wb") as f:
        pickle.dump(out, f)
