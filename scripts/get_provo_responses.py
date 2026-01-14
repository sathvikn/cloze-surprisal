import pickle, os, sys
import numpy as np
import pandas as pd

def read_indices(fn):
    my_dict = {}
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines[1:]:
            word, sentid, sentpos = line.strip().split(" ")
            my_dict[(int(sentid), int(sentpos))] = word
    return my_dict

def get_responses(df):
    # gets a list of unique responses
    all_responses = []
    for _, response in df.iterrows():
        all_responses += [response['Response']] * response['Response_Count']
    return all_responses

provo_path = "cloze_data/Provo_Corpus-Predictability_Norms.csv"
df = pd.read_csv(provo_path, encoding="latin-1")

output_dir = "cloze_data/context_responses"
responses = df.groupby(['Text_ID', 'Word_Number']).apply(get_responses)
responses = responses.reset_index().rename({0 : 'responses'}, axis = 1)

# START IGNORING
sent_info = df[['Text_ID', 'Sentence_Number', 'Word_Number', 'Response_Count']].drop_duplicates(['Text_ID', 'Sentence_Number', 'Word_Number'])
gb = sent_info.groupby('Text_ID')
sent_indices = {}
i = 0
for key, _df in gb:
    indices = list(_df.Sentence_Number)
    n = indices[-1]
    indices = np.array([1] + indices) + i - 1
    i += n
    sent_indices[key] = indices

df = df.drop_duplicates(['Text_ID'])
gb = df.groupby('Text_ID')
sents = []
word = []
sentid = []
docid = []
provo_textid = []
provo_wordnumber = []
for key, _df in gb:
    _word = _df.Text.iloc[0].split()
    _word = [x.replace('Õ', "'").replace('"', "'",) for x in _word]
    _sentid = sent_indices[key]
    if len(_sentid) == len(_word) - 1:
        if _word[0] == 'Voltaire':
            _sentid = list(_sentid)
            _sentid = [_sentid[0]] + _sentid
            _sentid = np.array(_sentid)
        else:
            _sentid = list(_sentid)
            _sentid.append(_sentid[-1])
            _sentid = np.array(_sentid)
    assert len(_word) == len(_sentid), 'Length mismatch. Got %s words and %d sentids' % (len(_word), len(_sentid))
    word.append(_word)
    sentid.append(_sentid)
    provo_textid.append([key] * len(_word))
    provo_wordnumber.append(np.arange(len(_word)) + 1)
    docid += ['d%s' % key] * len(_word)

word = np.concatenate(word, axis=0)
sentid = np.concatenate(sentid, axis=0)
provo_textid = np.concatenate(provo_textid)
provo_wordnumber = np.concatenate(provo_wordnumber)

out = pd.DataFrame(
    {
        'word': word,
        'docid': docid,
        'sentid': sentid,
        'text_id': provo_textid,
        'Word_Number': provo_wordnumber,
    }
)
out['sentpos'] = out.groupby(sentid).cumcount() + 1
out['tr'] = out.groupby(docid).cumcount()
out['startofsentence'] = (out.sentpos == 1).astype('int')
out['endofsentence'] = out.startofsentence.shift(-1).fillna(1).astype('int')
out = pd.merge(out, responses, left_on=['text_id', 'Word_Number'], right_on=['Text_ID', 'Word_Number'], how='left')

nan_indices = out[out['responses'].isna()].index
response_dict = {}
for i in range(len(out.index)):
    row = out.iloc[i]
    unique_responses = row['responses']
    if i in nan_indices:
        unique_responses = []
    response_dict[(int(row['sentid']), int(row['sentpos']))] = unique_responses

with open(os.path.join(output_dir, 'provo_responses.pickle'), 'wb') as f:
    pickle.dump(response_dict, f)