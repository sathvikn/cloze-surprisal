import pandas as pd
import numpy as np
import os

output_dir = "cloze_data/lm_input"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for corpus_name in ["provo", "ucl", "bk21cloze"]:
    filename = f"output/{corpus_name}.cloze_add1_50.itemmeasures"
    df = pd.read_table(filename, sep = " ")
    if corpus_name == "bk21cloze":
        df['responses_exist'] = df['clozeprob'] != np.inf
        df['num_responses'] = 90
    else:
        df['responses_exist'] = ~df['clozeprob'].isna()
        if corpus_name == "ucl":
            df['num_responses'] = df['total_responses']
        else: # for Provo
            df['num_responses'] = df['Total_Response_Count']
    print(f"writing {corpus_name}")
    df[['sentid', 'sentpos', 'word', 'num_responses', 'responses_exist']].to_csv(f"{output_dir}/{corpus_name}.tsv", sep = "\t", index = False)