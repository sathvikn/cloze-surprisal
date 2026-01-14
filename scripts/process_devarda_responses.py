import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from spellchecker import SpellChecker

devarda_folder = "cloze_data/devarda"

# using their preprocessing scripts to get unsmoothed cloze estimates
trgt_compregension = ['get off the ground suddenly', 'will not wait happily', 'child that is determined to do what it wants','small dogs with long ears', 'stuck through with a sharp instrument', 'an integrated human-machine system', 'at its peak of success', 'the faint color of her skin', 'sliding box', 'worried and puzzled']
punct_words = {'Ill':"I'll", # correct mistakes in punctuation/spacing/etc
 'cant':"can't",
 'couldnt':"couldn't",
 'didnt':"didn't",
 'doesnt':"doesn't",
 'dont':"don't",
 'friends':"friend's",
 'hes' : "he's",
 'mans':"man's",
 'shouldnt':"shouldn't",
 'wasnt':"wasn't",
 'wouldnt':"wouldn't",
 'youll':"you'll",
 'youre' : "you're"
 }

def clean_cloze_questions(series):
    qs = []
    for i, x in enumerate(series):
        q = re.sub("\[Field-1\]\.\.\.", "", x) # remove unnecessary string stuff
        q = re.sub("- Write the next word of the sentence:", "", q).strip()
        qs.append(q)
    print("N° items =", len(qs))
    return pd.Series(qs)
    
def clean_cloze_responses(df):
    cols = df.columns
    out = []
    corrected = df.copy(deep=True)
    logs = []
    spell = SpellChecker()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for col in cols:
            try:
                w = row[col].split(" ")[0].lower() # some participants wrote more than 1 word
                w = re.sub("’", "'", w)
                w_spell = spell.correction(w)
                if w_spell:
                    w_spell = w_spell
                else:
                    w_spell = w
                if w_spell != w:
                    logs.append([index, col, w, w_spell])
                corrected.loc[index, col] = w_spell
            except AttributeError: # empty string, only space
                pass
    logs = pd.DataFrame(logs, columns = ["idx", "Q_num", "word", "corrected"])
    #print("\n\nPercentage corrected =", round((logs.size/corrected.size)*100, 4), "%")
    print("\n\nCorrected ", logs.size, "out of", corrected.size, "words") 
    return corrected, logs

def get_cloze_unsmoothed(candidates, target):
    target = re.sub("[\.\,\?\!\:\;]", "", target)
    if target in punct_words.keys():
        target = punct_words[target]
    cloze_dict = {}
    N = float(len(candidates))
    for word in candidates:
        num_responses = candidates.count(word)
        cloze_dict[word] = num_responses / N
    if target in cloze_dict.keys():
        cloze_p = cloze_dict[target]
    else:
        cloze_p = 0
    cloze_surprisal = -np.log2(cloze_p) 
    return {
            'total_responses' : N,
            'word_responses' : candidates.count(target),
            'clozeprob_0': cloze_p,
            'clozesurp_0' : cloze_surprisal
            }

def load_clean_cp(listnum, items):
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
        target = list(items["word"])[n].strip()
        cloze_results.append(get_cloze_unsmoothed(candidates, target))
    return pd.DataFrame(cloze_results)

if __name__ == "__main__":

    devarda = pd.read_csv(f"{devarda_folder}/all_measures.csv")
    sentences = devarda.apply(lambda row : f"{row['item']} {row['word']}", axis = 1)

    item_set = pd.read_csv(f"{devarda_folder}/item-set.csv", dtype={ 'sentence':np.str_, 'word':np.str_, 'word2':np.str_})
    item_set["sentence"] = item_set["sentence"].str.strip()

    items = {i : item_set[item_set["list"] == i] for i in range(1, 9)}
    cloze = pd.concat([load_clean_cp(i, items[i]) for i in range(1, 9)], ignore_index=True) # combines the cloze responses in each list
    devarda = pd.concat([devarda, cloze], axis = 1) # merges the unsmoothed cloze results/response details with the original data
    devarda.to_csv("cloze_data/devarda_cloze_lm_test.csv", index = False)