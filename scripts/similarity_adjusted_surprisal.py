import sys
import pickle
import string
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from get_llm_surprisal import generate_stories

corpus_config = {
    "provo" : {"sentitems" : "text_data/provo.sentitems",
                "cloze" : "cloze_data/lm_input/provo.tsv", "responses": "cloze_data/provo_responses.pickle"},
    "ucl" : {"sentitems" : "text_data/ucl.sentitems",
             "cloze" : "cloze_data/lm_input/ucl.tsv", "responses": "cloze_data/ucl_devarda_responses.pkl"},
}

def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def cosine_sim(w, w_prime):
    # Meister et al use this formula
    return 0.5 * ((torch.dot(w, w_prime) / (torch.norm(w) * torch.norm(w_prime))) + 1)

def compute_sim_adj_surp(context, word, responses, model, tokenizer):
    word_indices = tokenizer.encode(f"{tokenizer.bos_token} {word}")[1:]
    word_embedding = model.transformer.wte.weight[word_indices].mean(dim = 0)
    response_weighted_probs = []
    word_values = {}
    for w_prime in responses:
        if w_prime not in word_values:
            response_prob = torch.prod(compute_word_prob(context, w_prime, model, tokenizer)).item()
            response_indices = tokenizer.encode(f"{tokenizer.bos_token} {w_prime}")[1:]
            response_embedding = model.transformer.wte.weight[response_indices].mean(dim = 0)
            cosine_to_word = cosine_sim(word_embedding, response_embedding).item()
            word_values[w_prime] = response_prob * cosine_to_word
        response_weighted_probs.append(word_values[w_prime])
    return -torch.log2(torch.tensor(sum(response_weighted_probs))).item()

def compute_word_prob(context, word, model, tokenizer):
    # regular surprisal, summing surprisals of multitoken words
    tokenizer_input = f"{tokenizer.bos_token} {context} {word}"
    lm_input = tokenizer(tokenizer_input, return_tensors="pt")
    outputs = model(**lm_input)
    logits = outputs.logits
    probs = torch.softmax(logits, dim = -1)
    word_ids = tokenizer.encode(f"{tokenizer.bos_token} {word}")[1:]
    preceding_indices = -1 - (len(word_ids) - 1)
    word_probs = probs[:, preceding_indices:, word_ids]
    return word_probs

def main():
    corpus_name = sys.argv[1]
    assert corpus_name in {"provo", "ucl"}
    corpus = corpus_config[corpus_name]
    print(f"Reading corpus and cloze data for {corpus_name}")

    stories = generate_stories(corpus['sentitems']) # get corpus data
    cloze_items = pd.read_table(corpus['cloze'], sep = "\t")
    response_path = corpus['responses']
    use_lm_responses = (len(sys.argv) == 4)
    if use_lm_responses:
        response_path = sys.argv[2]
        run_id = sys.argv[3]
    cloze_responses = read_pickle(response_path)

    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_size = "left")
    model = AutoModelForCausalLM.from_pretrained("gpt2") # can be adapted to other models
    model.eval()

    words = []
    lmsurp = []
    current_index = 0
    print(f"Processing {corpus_name}")
    for story in tqdm(stories):
        story_words = story.split(" ")
        words.extend(story_words)
        # for each word, generate a context, tokenize it, and compute similarity adjusted surprisal over the context's cloze responses
        context = ""
        for word in story_words:
            word_data = cloze_items.iloc[current_index]
            sentid, sentpos = word_data['sentid'], word_data['sentpos']
            print(context, word, sentid, sentpos)
            try:
                assert word == str(word_data['word'])
            except AssertionError:
                assert word == "None"
            responses_exist = word_data['responses_exist'].item()
            if responses_exist:
                # Compute similarity adjusted surprisal using the cloze responses
                responses = cloze_responses[(sentid, sentpos)]
                if use_lm_responses: # if working w LM completions, indexing with (sentid, sentpos) gets both LM responses and the context 
                    responses = responses['lm_completions']
                print(f"Computing similarity adjusted surprisal for cloze responses: {responses}")
                if len(responses) > 0:
                    surp = compute_sim_adj_surp(context, word, responses, model, tokenizer)
                else:
                    # No cloze data for this word in the sentence, compute regular surprisal of the word
                    print(f"No cloze responses, computing regular surprisal")
                    token_probs = compute_word_prob(context, word, model, tokenizer)
                    surp = torch.sum(-torch.log2(token_probs)).item()
                print(surp)
                lmsurp.append(surp)
            else: # no data for this sentence
                lmsurp.append("NaN")
            context += f" {word}"
            current_index += 1
    if use_lm_responses:
        output_file = f"output/{corpus_name}.gpt2_exp3_lm_{run_id}.itemmeasures"
    else:
        output_file = f"output/{corpus_name}.gpt2_exp3.itemmeasures"
    cloze_items['lmsurp'] = lmsurp
    cloze_items.to_csv(output_file, index = False, sep=' ', na_rep='NaN')

if __name__ == "__main__":
    main()