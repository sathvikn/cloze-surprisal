# cloze-surprisal

This is the repository for our paper "Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal (Nair* & Oh*, ACL 2026)".
Here, we calculate cloze- and LM-based surprisal for the three datasets (Provo; de Varda UCL; Brothers and Kuperberg).

Independently from this repository, the three datasets have also been integrated into the [(LME) regression toolkit from the Schuler group](https://github.com/modelblocks/modelblocks-release) (Modelblocks hereafter). 

We want to make two manipulations:

1) On the cloze surprisal side, we'd want to do 'due diligence' by checking whether different smoothing methods make a difference (i.e. we'd want to compare against the best version of cloze surprisal we can get).
2) On the LM surprisal side, we'd want to implement different hypotheses about what makes LM surprisal a better predictor of RT than cloze surprisal.

## General format
Modelblocks interfaces with space-delimited, two-column predictor files named `{dataset}.{identifier}.tokmeasures` (token level) or `{dataset}.{identifier}.itemmeasures` (word level). See under `output/` for GPT-2 surprisal examples. It is crucial that the tokens/words (i.e. the rows) are in this exact order for each dataset. This is relatively straightforward for LM surprisal, but maybe a bit tricky for cloze surprisal.

## LM surprisal
The surprisal calculation script `scripts/get_llm_surprisal.py` is the very basic version without WT decoding. With the hypotheses we have, adding WT decoding will not change the results we have significantly. It also has been adapted for Hypotheses 2 and 3 under Experiment 2.

The script will take as input the `{dataset}.sentitems` files under `text_data`. `!ARTICLE` separates the text such that they get fed into different LM context windows. The `{dataset}.sentitems` are ordered such that they yield the 'correct' order for the `tok/itemmeasures` files.

## Experiment 1: Cloze surprisal
It's a bit trickier to derive `itemmeasures` for cloze surprisal, as the exact way in which the cloze data is provided differs from dataset to dataset. Ideally, we'd have comparable `itemmeasures` files with `nan` for the values that are not provided (e.g. the very first word of each article, for which cloze data was not collected). I think I'll have to follow up with dataset-specific instructions for how the `.csv` files under `cloze_data` should be processed.

The code for preprocessing/extracting cloze data works slightly differently for each dataset:

Provo: `python3 scripts/process_provo.py cloze_data/Provo_Corpus-Predictability_Norms.csv  > foo`

- Cloze surprisal calculation takes place in this code. Provide the file name first, the smoothing parameter second, and the "estimated vocabulary" parameter third.

de Varda UCL: `python3 scripts/process_ucl.py text_data/ucl.indices cloze_data/devarda_cloze_lm.csv 2 50 > foo`

BK21: `python3 scripts/process_bk21cloze.py text_data/bk21cloze.indices cloze_data/SPRT_LogLin_216.csv 2 50> foo`

- For UCL, we first need to modify the `cloze_data/devarda/all_measures.csv` provided by De Varda et al. To generate `devarda_cloze_lm.csv`, run `python3 scripts/process_devarda_responses.py`. This reads the raw cloze responses in `cloze_data/devarda/cloze_responses`, computes unsmoothed cloze probabilities & responses counts for each context, and combines them with the provided measures.

To get all smoothed estimates for a corpus, run `python3 scripts/smoothed_clozesurp.py provo`, and `.itemmeasures` files with smoothed cloze probabilities & surprisals will show up in `output`. This file implements add-k smoothing, for k = 0, 0.1, 0.5, 1, 2, and 5, and estimated |V| sizes of 50, 100, and 200. These values can be adjusted in the variable `smoothing_params`.

## Experiment 2: Testing the Hypotheses
In order to make sure we are running the hypotheses on items that have existing cloze responses, we should load TSVs from the `cloze_data/lm_input` directory. These are generated with `python scripts/make_summaries.py`. 

Hypothesis 1 tries to put LM and cloze probability on equal footing by sampling a smaller number of completions from an LM. Run `python scripts/sample_completions.py [corpus name] [job ID]` to get these estimates. Outputs for each word are under `outputs/[corpus]_gpt2_h1_{job ID}.itemmeasures`, and the columns `lmprob` and `lmsurp` represent the measures from this manipulation, smoothed with k = 1 and V = 200. The job IDs distinguish different runs. Results are stored as pickle files in `output/h1_responses`.

Hypothesis 2 defines k-means clusters over GPT2's token embeddings, and adds the probabilities of the responses within the token's cluster. Run `python scripts/get_llm_surprisal.py text_data/[corpus_name].sentitems gpt2 clustering [number of clusters] [job ID] word`. We report results for k=20, 40, 80, 100, 500, and 1000 clusters, across five runs due to sample variation. The clustering results are saved as dictionaries that map from cluster IDs to tokens. They are stored as pickle files in `output/h2_clusters`.

Hypothesis 3 assumes that cloze responses are biased to more frequent words. Run `python scripts/get_llm_surprisal.py text_data/[corpus name].sentitems gpt2 freq 4 word` to get estimates for surprisal where we restrict the LM's vocabulary to tokens with a log frequency greater than 4 (as measured by `wordfreq`). This is around 1/3 of the GPT2 vocabulary. For tokens that are below the threshold, we assign probability 1 / (vocab size) + 1. We renormalize the other probabilities to reflect this.

### Experiment 3: Similarity-Adjusted Surprisal
In Experiment 3, we combine the sets of responses with their LM probabilities by computing similarity-adjusted surprisal. This is done by considering the cosine similarity of the non-contextual  embeddings of the word with the cloze completions, and weighting by the completions' surprisal, see [Meister et al. (2024)](https://aclanthology.org/2024.emnlp-main.921.pdf) for more details. For this experiment, we extract cloze responses for Provo and UCL with `scripts/get_provo_responses.py` and `scripts/get_ucl_responses.py`, respectively. These responses are dictionaries with `(sentid, sentpos)` as keys and a list of unique cloze responses as the values. They are saved as pickles in the `cloze_data` folder. After running these scripts, run `python scripts/similarity_adjusted_surprisal.py ucl` to compute similarity-adjusted surprisal for UCL.
We also compute similarity-adjusted surprisal over LM responses from Experiment 1. To get one set of results from UCL, run `python scripts/similarity_adjusted_surprisal.py ucl output/h1_responses/ucl_lm_responses.pickle`.