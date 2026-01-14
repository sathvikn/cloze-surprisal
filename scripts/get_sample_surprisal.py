import pickle
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sample_completions import compute_lm_cloze
from get_llm_surprisal import generate_stories

corpus_config = {
    "provo" : {"sentitems" : "text_data/provo.sentitems",
                "cloze" : "cloze_data/lm_input/provo.tsv", "default_n" : 40},
    "ucl" : {"sentitems" : "text_data/ucl.sentitems",
             "cloze" : "cloze_data/lm_input/ucl.tsv", "default_n" : 80},
    "bk21cloze" : {"sentitems" : "text_data/bk21cloze.sentitems",
                   "cloze" : "cloze_data/lm_input/bk21cloze.tsv", "default_n" : 90}
}

def main():
    corpus_name = sys.argv[1]
    corpus = corpus_config[corpus_name]
    job_id = sys.argv[2]
    k = int(sys.argv[3])
    V = int(sys.argv[4])

    print(f"Reading corpus and cloze data for {corpus_name} (job {job_id})")
    stories = generate_stories(corpus['sentitems']) # get corpus data
    cloze_data = pd.read_table(corpus['cloze'], sep = "\t")

    with open(f"output/{corpus_name}_lm_responses_{job_id}.pickle", "rb+") as f:
        all_completions = pickle.load(f)
    
    words = []
    lmprob = []
    lmsurp = []
    lm_response_count = []
    # all_completions = {}
    current_index = 0

    print(f"Processing {corpus_name}")
    for story in tqdm(stories):
        story_words = story.split(" ")
        words.extend(story_words)
        # for each word, generate a context, tokenize it, and sample completions. Then compute cloze surprisal over k tokens
        context = ""
        for word in story_words:
            word_data = cloze_data.iloc[current_index]
            sentid, sentpos = word_data['sentid'], word_data['sentpos']
            print(context, word, sentid, sentpos)
            try:
                assert word == str(word_data['word'])
            except AssertionError:
                assert word == "None"
            sample_context = word_data['responses_exist'].item()
            # this excludes words without cloze responses for De Varda & BK21
            if (sentid, sentpos) in all_completions:
                ctx_completions = all_completions[(sentid, sentpos)]
                lm_cloze = compute_lm_cloze(ctx_completions["lm_completions"], word.strip(), k, V)
                lm_response_count.append(len(ctx_completions["lm_completions"]))
                lmprob.append(lm_cloze)
                lmsurp.append(-np.log2(lm_cloze))
            else:
                lmprob.append("NaN")
                lmsurp.append("NaN")
                lm_response_count.append("NaN")
            context += f" {word}"
            current_index += 1
    output_file = f"output/{corpus_name}.gpt2_h1_{job_id}.itemmeasures"
    
    cloze_data['lmprob'] = lmprob
    cloze_data['lmsurp'] = lmsurp
    cloze_data['lm_responses'] = lm_response_count
    cloze_data.to_csv(output_file, index = False, sep=' ', na_rep='NaN')

if __name__ == "__main__":
    main()