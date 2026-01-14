"""
Calculates LLM surprisal from the following LLM families:

GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"

Pythia family:
"EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
"EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
each with checkpoints specified by training steps:
"step1", "step2", "step4", ..., "step142000", "step143000"
"""

import os, pickle, sys, tqdm, torch, transformers, wordfreq
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast, GPTNeoXForCausalLM


def generate_stories(fn):
    stories = []
    f = open(fn)
    first_line = f.readline()
    assert first_line.strip() == "!ARTICLE"
    curr_story = ""

    for line in f:
        sentence = line.strip()
        if sentence == "!ARTICLE":
            stories.append(curr_story[:-1])
            curr_story = ""
        else:
            curr_story += line.strip() + " "

    stories.append(curr_story[:-1])
    return stories

def find_high_freq_tokens(tokenizer, cutoff):
    vocab = tokenizer.get_vocab()
    return [tokenizer.convert_tokens_to_ids(token) for token in vocab.keys() 
                if wordfreq.zipf_frequency(token.strip("Ġ"), "en") >= cutoff]

def get_token_clusters(model, num_clusters):
    # kmeans for now
    print(f"Running k-means with {num_clusters} clusters")
    input_embeddings = model.transformer.wte.weight
    kmeans = KMeans(n_clusters=num_clusters)
    emb_arr = np.array(input_embeddings.tolist())
    kmeans.fit(emb_arr)
    cluster_assignments = kmeans.predict(emb_arr)
    clusterid_to_tokens = {
        int(i) : np.where(cluster_assignments == i)
        for i in np.arange(num_clusters)
    }
    tokenid_to_clusterid = {
                            int(i) : int(cluster_assignments[i]) 
                            for i in np.arange(len(cluster_assignments))
                            }
    return clusterid_to_tokens, tokenid_to_clusterid

def save_clustering(cluster_tokens, corpus_name, num_clusters, run_id):
    filename = f"output/h2_clusters/{corpus_name}_{num_clusters}_clusters_{run_id}.pickle"
    with open(filename, 'wb') as clusters:
        pickle.dump(cluster_tokens, clusters)

def main():
    sentitem_path = sys.argv[1]
    corpus_name = sentitem_path.split("/")[1].split(".")[0]
    stories = generate_stories(sentitem_path)
    model_variant = sys.argv[2].split("/")[-1]
    manipulation = sys.argv[3]
    hparam = int(sys.argv[4]) # for semantics, it's number of clusters; for frequency, it's the cutoff
    run_id = sys.argv[-2]
    mode = sys.argv[-1]
    assert mode in {"token", "word"}, ValueError('Calculation mode must be "token" or "word"')
    assert manipulation in {"cluster", "freq"}, ValueError('Manipulation must either "cluster" or "freq"')

    if "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(sys.argv[2])
    elif "gpt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    elif "opt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    elif "pythia" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], revision=sys.argv[3])
    else:
        raise ValueError("Unsupported LLM variant")

    if "pythia" in model_variant:
        model = GPTNeoXForCausalLM.from_pretrained(sys.argv[2], revision=sys.argv[3])
    else:
        model = AutoModelForCausalLM.from_pretrained(sys.argv[2])

    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    high_freq_tokens = []
    if manipulation == "freq":
        high_freq_tokens = find_high_freq_tokens(tokenizer, hparam)
        base_prob = torch.tensor(1 / (len(high_freq_tokens) + 1))
        smoothing_factor = len(high_freq_tokens) / (len(high_freq_tokens) + 1)

    cluster2tokens, token2cluster = {}, {}
    if manipulation == "cluster":
        cluster2tokens, token2cluster = get_token_clusters(model, hparam)
    
    save_clustering(cluster2tokens, corpus_name, hparam, run_id)

    batches = []
    words = []

    print(f"tokenizing stories for {corpus_name}")
    for story in stories:
        words.extend(story.split(" "))
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask

        # these tokenizers do not append bos_id by default
        if "gpt" in model_variant or "pythia" in model_variant:
            ids = [bos_id] + ids
            attn = [1] + attn

        start_idx = 0

        # sliding windows with 50% overlap
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:
            # for models that explicitly require the first dimension (batch_size)
            if "gpt-neox" in model_variant or "pythia" in model_variant or "opt" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
                                                            "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}),
                                torch.tensor(ids[1:ctx_size+1]), start_idx, True))
            # for other models
            elif "gpt" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]),
                                                            "attention_mask": torch.tensor(attn[:ctx_size])}),
                                torch.tensor(ids[1:ctx_size+1]), start_idx, True))

            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)

        # remaining tokens
        if "gpt-neox" in model_variant or "pythia" in model_variant or "opt" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:-1]).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn[:-1]).unsqueeze(0)}),
                           torch.tensor(ids[1:]), start_idx, False))
        elif "gpt" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:-1]),
                                                        "attention_mask": torch.tensor(attn[:-1])}),
                           torch.tensor(ids[1:]), start_idx, False))

    curr_word_prob = []
    curr_word_surp = []
    curr_toks = []
    curr_word_ix = 0
    is_continued = False

    all_word_prob = []
    all_word_surp = []
    all_words = []
    is_multitoken = []

    print("computing surprisal")
    for batch in tqdm.tqdm(batches):
        batch_input, output_ids, start_idx, will_continue = batch
        with torch.no_grad():
            model_output = model(**batch_input)
        toks = tokenizer.convert_ids_to_tokens(output_ids)
        index = torch.arange(0, output_ids.shape[0])

        if manipulation == "cluster":
            prob, surp = [], []
            probs = softmax(model_output.logits).squeeze(0)
            for i in range(len(output_ids)):
                output_token = output_ids[i]
                context_probs = probs[i]
                target_cluster = token2cluster[output_token.item()]
                cluster_tokens = cluster2tokens[target_cluster]
                item_probs = context_probs[cluster_tokens]
                cluster_prob = torch.sum(item_probs)
                prob.append(cluster_prob)
                surp.append(-torch.log2(cluster_prob))
                
        elif manipulation == "freq":
            prob = []
            surp = []
            hf_logits = model_output.logits[:, high_freq_tokens].squeeze()
            probs = softmax(hf_logits)
            for i in range(len(toks)):
                token = output_ids[i]
                if token in high_freq_tokens:
                    vocab_index = high_freq_tokens.index(token)
                    token_prob = probs[i][vocab_index] * smoothing_factor
                    token_prob = token_prob
                else:
                    token_prob = base_prob
                prob.append(token_prob)
                surp.append(-torch.log2(torch.tensor(token_prob)))
        else: 
            prob = softmax(model_output.logits).squeeze(0)[index, output_ids]
            surp = -1 * torch.log2(prob)

        if mode == "token":
            # token-level surprisal
            for i in range(start_idx, len(toks)):
                cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
                print(cleaned_tok, surp[i].item())

        elif mode == "word":
            # word-level surprisal
            # if the batch starts a new story
            if not is_continued:
                curr_word_surp = []
                curr_word_prob = []
                curr_toks = []
            for i in range(start_idx, len(toks)):
                curr_word_prob.append(prob[i].item())
                curr_word_surp.append(surp[i].item())
                curr_toks += [toks[i]]
                curr_toks_str = tokenizer.convert_tokens_to_string(curr_toks)
                # summing token-level surprisal
                if words[curr_word_ix] == curr_toks_str.strip():
                    all_words.append(curr_toks_str.strip())
                    all_word_prob.append(torch.prod(torch.tensor(curr_word_prob)).item())
                    all_word_surp.append(sum(curr_word_surp))
                    is_multitoken.append(len(curr_toks) > 1)
                    curr_word_prob = []
                    curr_word_surp = []
                    curr_toks = []
                    curr_word_ix += 1

        is_continued = will_continue

        del model_output

    out = pd.DataFrame({
        'word' : all_words,
        'lmprob' : all_word_prob,
        'lmsurp' : all_word_surp,
        'is_multitoken' : is_multitoken
    })
    
    if manipulation == "cluster":
        filename = f"output/{corpus_name}.gpt2_h3_{hparam}_clusters_{run_id}.itemmeasures"
        out.to_csv(filename, index = False, sep = " ")
    if manipulation == "freq":
        filename = f"output/{corpus_name}.gpt2_h4_above_{hparam}.itemmeasures"
        out.to_csv(filename, index = False, sep = " ")

if __name__ == "__main__":
    main()
