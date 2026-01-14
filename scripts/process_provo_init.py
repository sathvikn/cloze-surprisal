import sys
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

def get_cloze(df):
    sel = df.Word == df.Response
    df = df[sel]
    if len(df):
        return df.Response_Proportion.sum()
    return 0.

def get_count(df):
    return df.Total_Response_Count.max()

df = pd.read_csv(sys.argv[1], encoding="latin-1")

# get proportions of responses that match the actual word (cloze probs)
cloze = df.groupby(['Text_ID', 'Word_Number']).apply(get_cloze)
# get total number of human responses (i.e. the denominator) for a given prefix
count = df.groupby(['Text_ID', 'Word_Number']).apply(get_count)
cloze = pd.concat([cloze, count], axis=1).reset_index().rename(lambda x: 'clozeprob' if x == 0 else 'Total_Response_Count' if x == 1 else x, axis=1)

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
out = pd.merge(out, cloze, left_on=['text_id', 'Word_Number'], right_on=['Text_ID', 'Word_Number'], how='left')

# STOP IGNORING
# filling undefined values; 0. for undefined cloze probs, 40 for undefined counts
clozeprob = out.clozeprob
clozeprob = np.where(clozeprob.isna(), 0., clozeprob)
out.clozeprob = clozeprob
count = out.Total_Response_Count
count = np.where(count.isna(), 40, count).astype(int)
out.Total_Response_Count = count
count = out.Total_Response_Count
# apply smoothing to cloze prob, then convert into cloze surp
# looks like add-1 smoothing
gt0 = (out.clozeprob <= 0).astype(float)
smoothing_factor = count / (count + 1)
base_prob = gt0 / (count + 1)
out.clozeprob = out.clozeprob * smoothing_factor + base_prob
out['clozesurp'] = -np.log2(out.clozeprob)

name2discid = {x: i for i, x in enumerate(sorted(list(out['docid'].unique())))}
out['discid'] = out.docid.map(name2discid)
out['discpos'] = out.groupby('discid').cumcount() + 1
out = out.sort_values(by=['discid', 'discpos'], ascending=[True, True])

del out['Text_ID']

out[["word", "sentid", "sentpos", "Total_Response_Count", "clozeprob", "clozesurp"]].to_csv(sys.stdout, index=False, sep=' ', na_rep='NaN')