# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:41:14 2020

@author: truthless
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os.path as osp
import json
sys.path.append("/home/mawenchang/PROREC-Torch")
from data.utils import da_tree_serial,utter_lexical_redial,utter_lexical_gorecdial
from data.redial import ReDial
from data.gorecdial import GoRecDial
from termcolor import colored
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

n_gpu=torch.cuda.device_count()
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}

            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


set_seed(42)

class Generator():
    def __init__(self,conf):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        self.conf=conf
        self.tokenizer = tokenizer_class.from_pretrained(conf['gpt_path'])
        self.model= model_class.from_pretrained(conf['gpt_path'])
        self.model.to(device)
        self.model.eval()


    def generate(self,raw_text):
        stop_token='<|endoftext|>'
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        length=self.conf['max_length']
        if length < 0 and self.model.config.max_position_embeddings > 0:
            length = self.model.config.max_position_embeddings
        elif 0 < self.model.config.max_position_embeddings < length:
            length = self.model.config.max_position_embeddings  # No generation bigger than model size 
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop

        #logger.info(args)

        context_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)
        out = sample_sequence(
            model=self.model,
            context=context_tokens,
            num_samples=1,#args.num_samples
            length=self.conf['max_length'],
            temperature=self.conf['temperature'],
            top_k=self.conf['top_k'],
            top_p=self.conf['top_p'],
            device=device,
        )
        out = out[:, len(context_tokens):].tolist()
        for o in out:
            text = self.tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(stop_token) if stop_token else None]

            #print(text)

        return text


if __name__ == '__main__':
    root=osp.dirname(osp.dirname(osp.abspath(__file__)))
    save_path=osp.join(root,"saved",'gorecdial_DA.json')
    data_path=osp.join(root,"data","gorecdial")
    #data_path=osp.join(root,"data","redial")
    generate_path=osp.join(root,"saved","gorecdial_gen.json")
    #context_path=osp.join(root,"saved","context.json")

    #dataset=ReDial(data_path,flag="test")
    dataset=GoRecDial(data_path,flag="test")
    #f=open(generate_path)
    #generate_all=json.load(f)
    generate_all={}
    #print(len(dataset))
    f=open(save_path)
    das=json.load(f)
    #print(len(das))
    #with open('/home/mawenchang/PROREC-Torch/data/id2name_redial.json', 'r') as f:
    with open('/home/mawenchang/PROREC-Torch/data/id2name_gorecdial.json', 'r') as f:
        id2name = json.load(f)
    #with open('/home/mawenchang/PROREC-Torch/data/mid2name_redial.json', 'r') as f:
    with open('/home/mawenchang/PROREC-Torch/data/mid2name_gorecdial.json', 'r') as f:
        mid2name = json.load(f)
    # f=open(generate_path)
    # gens=json.load(f)
    # for key,value in gens.items():
    #       gens[key]['KBRD']=value['KBRD'].lower()
    #       gens[key]['DCR']=value['DCR'].lower()
    #       gens[key]['Human']=value['Human'].lower()
    #       gens[key]['REDIAL']=value['REDIAL'].lower()
    # #     gens[key]['KBRD']=utter_lexical_redial_kbrd(value['KBRD'],mid2name)
    # #     #input()
    # f=open(generate_path,'w')
    # json.dump(gens,f)



    #-----------------------------------------------------
    # root=osp.dirname(osp.dirname(osp.abspath(__file__)))
    # save_path=osp.join(root,"saved",'redial_DA_new.json')
    # data_path=osp.join(root,"data","redial")
    # generate_path=osp.join(root,"saved","redial_gen_intent.json")

    
    # dataset=ReDial(data_path,flag="test")
    # #f=open(generate_path)
    # #generate_all=json.load(f)
    # generate_all={}
    # print(len(dataset))
    # f=open(save_path)
    # das=json.load(f)
    # #print(len(das))
    # with open('/home/mawenchang/PROREC-Torch/data/id2name_redial.json', 'r') as f:
    #     id2name = json.load(f)
    # with open('/home/mawenchang/PROREC-Torch/data/mid2name_redial.json', 'r') as f:
    #     mid2name = json.load(f)
    # context={}
    for i,item in enumerate(das):
    #     con=dataset[i].dialog_history
    #     for j in range(len(con)):
    #         con[j]=utter_lexical_redial(con[j],mid2name)
    #     key=dataset[i].my_id
    #     context[key]=con
    #     print(key)
    #     print(con)
    #     input()
    
    # f=open(context_path,'w')
    # json.dump(context,f)
        if item['key'] in generate_all.keys():
            #data=generate_all[item['key']]
            context=dataset[i].dialog_history
            #for turn in context:
                #print(utter_lexical_redial(turn,mid2name))
            #print(colored(utter_lexical_redial(data['label'],mid2name),'green'))
            #print(colored(data['generated'],'red'))
            #input()
            #continue
        #print(item['key'])
        #print(dataset[i].my_id)
        DA=da_tree_serial(item,id2name)
        #context=""
        #print(dataset[i].dialog_history)
        if len(dataset[i].dialog_history)!=0:
            context=dataset[i].dialog_history[-1]
        else:
            context="hello"
        #context=utter_lexical_redial(context,mid2name)
        context=utter_lexical_gorecdial(context,mid2name)
        gpt_in=context+" @ "+DA+" &"
        #gpt_in=context+"@"+item['intent']+" &" 
        print(gpt_in.lower())
        generated=generate(gpt_in.lower())
        print(utter_lexical_gorecdial(dataset[i].oracle_response,mid2name))
        cur_turn={"generated":generated,"label":utter_lexical_gorecdial(dataset[i].oracle_response,mid2name)}
        generate_all[item['key']]=cur_turn
        if (i+1) % 50 ==0:
            print("saving results...")
            with open(generate_path,'w') as f:
                json.dump(generate_all,f)
    print("saving results...")
    with open(generate_path,'w') as f:
        json.dump(generate_all,f)
