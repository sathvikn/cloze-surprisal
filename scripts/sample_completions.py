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

sample_tokens = 4 # sample 4 tokens, consider words up to 3 tokens.
k = 1
V = 50 # can be set thru input as well
# dictionary with path to stories
corpus_config = {
    "provo" : {"sentitems" : "text_data/provo.sentitems",
                "cloze" : "cloze_data/lm_input/provo.tsv", "default_n" : 40},
    "ucl" : {"sentitems" : "text_data/ucl.sentitems",
             "cloze" : "cloze_data/lm_input/ucl.tsv", "default_n" : 80},
    "bk21cloze" : {"sentitems" : "text_data/bk21cloze.sentitems",
                   "cloze" : "cloze_data/lm_input/bk21cloze.tsv", "default_n" : 90}
}

def get_next_token_generations(context_ids, attn, model, tokenizer, num_tokens, num_samples):
    input_tokens = context_ids.repeat(num_samples, 1)
    input_attn_mask = attn.repeat(num_samples, 1)
    model_input = BatchEncoding({'input_ids' : input_tokens.long(), 'attention_mask' : input_attn_mask.long()})        
    lm_samples = model.generate(**model_input, max_new_tokens=num_tokens, do_sample=True, num_beams=1)
    # multinomial sampling, probably the most naive method?
    outputs = tokenizer.batch_decode(lm_samples[:,len(context_ids):], skip_special_tokens=True)
    return [completion.strip().split(" ")[0] for completion in outputs] # get the first word

def compute_lm_cloze(sampled_words, correct_word, k, V):
    counts = Counter(sampled_words)
    N = len(sampled_words)
    return (counts[correct_word] + k) / (N + (k * V))

def decode_tokens(tokenizer, completions, is_multitoken = False):
    # Decode each token sequence, filtering out the bos token
    if is_multitoken:
        return [tokenizer.decode([token for token in token_seq if token != tokenizer.bos_token_id]).strip()
                                for token_seq in completions]
    return [tokenizer.decode(token_seq).strip()
                                for token_seq in completions]

def sample_ahead(model, lm_input):
    # run the model over the input and sample once from each context's next-token distribution
    next_token_outputs = model(**lm_input)
    next_token_logits = next_token_outputs.logits[:,-1:,]
    post_sampling_probs = torch.softmax(next_token_logits, dim = -1).squeeze(dim = 1)
    return torch.multinomial(post_sampling_probs, 1).squeeze()

def sample(context, model, tokenizer, bow_punct_tokens, num_tokens, num_samples):
    sampled_words = []
    tokenizer_output = tokenizer(context)
    ids = tokenizer_output.input_ids
    attn = tokenizer_output.attention_mask
    ids = [model.config.bos_token_id] + ids
    attn = [1] + attn
    ids, attn = torch.Tensor(ids), torch.Tensor(attn)
    lm_input = BatchEncoding({'input_ids' : ids.long(), 'attention_mask' : attn.long()})
    # Sampling the first set of tokens
    outputs = model(**lm_input)
    next_word_probs = torch.softmax(outputs.logits[-1], dim = -1)
    candidate_tokens = torch.multinomial(next_word_probs, num_samples, replacement = True)

    # Sampling the continuations.
    ids, attns = ids.repeat(num_samples, 1), attn.repeat(num_samples, 1)
    lm_input['input_ids'] = torch.cat((ids, candidate_tokens.reshape(-1, 1)), dim = 1).long()
    lm_input['attention_mask'] = torch.cat((attns, torch.ones(len(attns), 1)), dim = 1).long()
    next_samples = sample_ahead(model, lm_input)
    # if the next sampled token is eow, we add to sampled_words
    finished = torch.isin(next_samples, bow_punct_tokens) 
    sampled_words += decode_tokens(tokenizer, candidate_tokens[finished])
    samples_drawn = 2
    print(sampled_words, len(sampled_words))

    while len(sampled_words) < num_samples and samples_drawn < num_tokens:
        # getting indices for tokens that need more samples
        remaining_indices = torch.argwhere(~finished).squeeze()
        candidate_tokens = next_samples[~finished]

        # modifying the context from the LM input
        context_tokens = lm_input['input_ids'][remaining_indices]
        context_attentions = lm_input['attention_mask'][remaining_indices]

        if len(candidate_tokens) > 1:
            lm_input['input_ids'] = torch.cat((context_tokens, candidate_tokens.reshape(-1, 1)), dim = 1).long()
            lm_input['attention_mask'] = torch.cat((context_attentions, torch.ones(len(candidate_tokens), 1)), dim = 1).long()
        else:
            lm_input['input_ids'] = torch.cat((context_tokens, candidate_tokens)).long()
            lm_input['attention_mask'] = torch.cat((context_attentions, torch.ones(1))).long()
        
        # sampling the next token
        next_samples = sample_ahead(model, lm_input)
        finished = torch.isin(next_samples, bow_punct_tokens)

        concat_tokens = samples_drawn - 1 # index of how far to look back
        multitoken_indices = torch.cat((context_tokens[finished][:,-concat_tokens:], 
                                      candidate_tokens[finished].reshape(-1, 1)), dim = 1)
        multitoken_words = decode_tokens(tokenizer, multitoken_indices, is_multitoken=True)
        print(multitoken_words)
        sampled_words += multitoken_words
        samples_drawn += 1
        print(len(sampled_words), samples_drawn)
    
    # for token sequences that didn't finish getting sampled
    if not finished.all():
        if next_samples.dim():
            remaining_completions = torch.cat((context_tokens[~finished][:,-(samples_drawn - 1):], candidate_tokens[~finished].reshape(-1, 1)), dim = 1)
            other_seqs = decode_tokens(tokenizer, remaining_completions, is_multitoken=True)
        else: # one token didn't finish
            current_token = candidate_tokens.item()
            tokens = context_tokens[-(samples_drawn - 1):].tolist() + [current_token]
            other_seqs = [tokenizer.decode(tokens).strip()]
        sampled_words += other_seqs
        print(other_seqs, len(sampled_words))
    return sampled_words

def find_eow_tokens(vocab):
    # This is used to check if the next token sampled marks the end of a word - either starts with a bow token or is punctuation.
    bow_punct, without_bow = [], []
    for key, value in vocab.items():
        contains_punct = all([c in string.punctuation for c in key])
        if key[0] == "Ġ" or contains_punct:
            bow_punct.append(value)
        else:
            without_bow.append(value)
    return torch.tensor(bow_punct)

def main():
    corpus_name = sys.argv[1]
    corpus = corpus_config[corpus_name]
    job_id = sys.argv[2]
    print(f"Reading corpus and cloze data for {corpus_name} (job {job_id})")
    stories = generate_stories(corpus['sentitems']) # get corpus data
    cloze_data = pd.read_table(corpus['cloze'], sep = "\t")

    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_size = "left")
    model = AutoModelForCausalLM.from_pretrained("gpt2") # can be adapted to other models
    model.eval()

    eow_marking_tokens = find_eow_tokens(tokenizer.vocab)

    words = []
    lmprob = []
    lmsurp = []
    lm_response_count = []
    all_completions = {}
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
            if sample_context:
                print(f"Generating LM responses for WORD={word} given CONTEXT={context}")
                cloze_completions = int(word_data['num_responses'].item())
                if not cloze_completions:
                    cloze_completions = corpus_config[corpus_name]['default_n']
                ctx_completions = sample(context, model, tokenizer, eow_marking_tokens, sample_tokens, cloze_completions)
                all_completions[(sentid, sentpos)] = {"context": context, "lm_completions" : ctx_completions}
                lm_cloze = compute_lm_cloze(ctx_completions, word.strip(), k, V)
                lm_response_count.append(len(ctx_completions))
                lmprob.append(lm_cloze)
                lmsurp.append(-np.log2(lm_cloze))
            else:
                lmprob.append("NaN")
                lmsurp.append("NaN")
                lm_response_count.append("NaN")
            context += f" {word}"
            current_index += 1
    output_file = f"output/{corpus_name}.gpt2_h1_{job_id}.itemmeasures"
    lm_response_file = f'/fs/clip-scratch/sathvik/{corpus_name}_lm_responses_{job_id}.pickle'
    print(f"Finished processing, writing responses to {lm_response_file} and outputs to {output_file}")
    cloze_data['lmprob'] = lmprob
    cloze_data['lmsurp'] = lmsurp
    cloze_data['lm_responses'] = lm_response_count
    cloze_data.to_csv(output_file, index = False, sep=' ', na_rep='NaN')
    with open(lm_response_file, 'wb') as lm_responses:
        pickle.dump(all_completions, lm_responses)

if __name__ == "__main__":
    main()

