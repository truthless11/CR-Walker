import re
import numpy as np
from nltk.translate import bleu_score as nltkbleu
from collections import Counter
from nltk.util import ngrams

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
#    def remove_articles(text):
#        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    #return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_punc(lower(s)))

def bleu(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    try:
        return nltkbleu.sentence_bleu(
            [normalize_answer(a).split(" ") for a in answers],
            normalize_answer(guess).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method7,
        )
    except:
        return 0

def prec_recall_f1_score(pred_items, gold_items):
    """
    Computes precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def dist_str(n, hyps):
    #print(hyps)
    #print(n)
    all_scores = []
    for hyp in hyps:
        if '<EOS>' in hyp:
            index = hyp.index('<EOS>')
            hyp = hyp[:index]
        #print(hyp)
        sentence = hyp.split()
        #sentence = sentence.split(" ")
        

        distinct_ngrams = set(ngrams(sentence, n))
        #print(distinct_ngrams)
        if len(list(ngrams(sentence, n)))!=0:
            score = len(distinct_ngrams) / len(list(ngrams(sentence, n)))
            all_scores.append(score)

        #print(score)
        
        #input()

    return all_scores



def generate_n_grams(x, n):
    n_grams = set(zip(*[x[i:] for i in range(n)]))
    # print(x, n_grams)
    # for n_gram in n_grams:
    #     x.append(' '.join(n_gram))
    return n_grams


def distinct_n_grams(tokenized_lines, n):

    n_grams_all = set()
    for line in tokenized_lines:
        n_grams = generate_n_grams(line, n)
        # print(line, n_grams)
        n_grams_all |= n_grams
    total_len=0
    for item in tokenized_lines:
        total_len+=len(item)

    return len(set(n_grams_all)), len(set(n_grams_all)) / total_len#len(tokenized_lines)


if __name__ == '__main__':
    import json
    import os.path as osp
    root=osp.dirname(osp.dirname(osp.abspath(__file__)))
    #path=osp.join(root,"saved","all_gen.json")
    path=osp.join(root,"saved","redial_gen_e2e.json")  
    with open(path, 'r') as f:
        data_gen = json.load(f)
    
    with open('mid2name_redial.json','r') as f:
    #with open('mid2name_gorecdial.json','r') as f:
        mid2name=json.load(f)
    a=[]

    #model='CR_Walker'
    # all_lines=[[] for _ in range(1341)]
    # #line=[]
    # for key,value in data.items():
    #     dialog_num=int(key.split("_")[0])
    #     all_lines[dialog_num].append(value[model].strip())

    #lines = [item[model].strip() for item in data.values()]
    #data = [[item['Human'].strip(), item[model].strip()] for item in data.values()]
    lines = [item['generated'].strip() for item in data_gen.values()]
    # data=[]
    # for item in data_gen.values():
    #     gen=item['generated'].split("&")
    #     generate=""
    #     for sent in gen:
    #         rep=sent.replace("@"," ")
    #         if "(" not in rep and ")" not in rep:
    #             generate=rep
    #     data.append([item['label'].strip(),generate.strip()])

    bleu_array = []
    f1_array = []
    
    k=0
    for item in data_gen.values():
        k+=1
        ground, generate = [item['label']], item['generated']
        bleu_array.append(bleu(generate, ground))
        f1_array.append(f1_score(generate, ground))
    print("BLEU:",np.mean(bleu_array))
    print("F1:",np.mean(f1_array))


    tokenized = [line.split() for line in lines]


    for n in range(1, 6):
        cnt, percent = distinct_n_grams(tokenized, n)
        print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')
    