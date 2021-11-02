import sys
from transformers import BertModel,BertTokenizer
from nltk import word_tokenize
import torch
import torch.nn as nn
import os.path as osp
import json

from torch_geometric.data import DataLoader



root=osp.dirname(osp.dirname(osp.abspath(__file__)))
path=osp.join(root,"data","redial")


class Utterance_Embedder(nn.Module):
    def __init__(self,rnn_type="RNN_TANH",use_bert=True,rnn_hidden=64,dropout=0.5,num_turns=10,num_words=30,word_net=False):
        super(Utterance_Embedder,self).__init__()   
        model_class=BertModel 
        pretrained_weights = 'bert-base-uncased'
        self.rnn_hidden=rnn_hidden
        #self.maxlen=maxlen
        self.num_turns=num_turns
        self.num_words=num_words
        self.word_net=word_net
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        #self.device = torch.device('cuda:1')
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(768, rnn_hidden,dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(768, rnn_hidden, nonlinearity=nonlinearity, dropout=dropout)
        
        self.rnn_type=rnn_type
        self.key2index=json.load(open(osp.join(root,"data","key2index_3rd.json"),encoding='utf-8'))

    def forward(self,tokenized,length,max_len,init_hidden):
        bert_embed=self.model(**tokenized)[0].view(-1,self.num_turns,max_len,768).permute(1,0,2,3)#(num_turns,batch_size,max_len,768)
        sentence_rep=bert_embed[:,:,0,:]
        hidden=init_hidden

        output,hidden=self.rnn(sentence_rep,hidden)
        output=output[-1,:,:]

        return output


    def prepare_data(self,dialog_history,device,raw_history=False):
        pad_history=[]
        word_index=[]
        word_batch_index=[]
        all_words=[]
        for i in range(len(dialog_history)):
            turn_words=[]
            cur_history=""
            turns=len(dialog_history[i])
            if turns>self.num_turns:
                padded=dialog_history[i][turns-self.num_turns:]
                pad_history=pad_history+padded
                for sen in padded:
                    cur_history=cur_history+sen+" "
            elif turns<self.num_turns:
                pad=[]
                for j in range(self.num_turns-turns):
                    pad.append("")
                padded=pad+dialog_history[i]
                for sen in dialog_history[i]:
                    cur_history=cur_history+sen+" "

                pad_history=pad_history+padded
            else:
                pad_history=pad_history+dialog_history[i]
                for sen in dialog_history[i]:
                    cur_history=cur_history+sen+" "
            
            token_history=word_tokenize(cur_history)
            if len(token_history)>self.num_words:
                token_history=token_history[-self.num_words:]
            elif len(token_history)==0:
                word_index.append(0)
                word_batch_index.append(i)

            for word in token_history:
                word_index.append(self.key2index.get(word.lower(),0))
                word_batch_index.append(i)
                turn_words.append(self.key2index.get(word.lower(),0))
            all_words.append(turn_words)

        tokenized_dialog=self.tokenizer.batch_encode_plus(pad_history,pad_to_max_length=True,return_tensors="pt",)#[turn1,turn2,turn3....]
        for key in tokenized_dialog.keys():
            tokenized_dialog[key]=tokenized_dialog[key].to(device=device)
        all_length=torch.sum(tokenized_dialog["attention_mask"],dim=-1).view(-1,self.num_turns).permute(1,0)
        maxlen=tokenized_dialog["attention_mask"].size()[-1]
        batch_size=len(dialog_history)

        if self.rnn_type=="LSTM":
            init_hidden=(torch.zeros(1,batch_size,self.rnn_hidden),torch.zeros(batch_size,1,self.rnn_hidden)).to(device=device)
        else:
            init_hidden=torch.zeros(1,batch_size,self.rnn_hidden).to(device=device)

        w_index=None
        w_batch_index=None
      
        if self.word_net:
            w_index=torch.Tensor(word_index).long().to(device=device)
            w_batch_index=torch.Tensor(word_batch_index).long().to(device=device)
        if raw_history:
            return tokenized_dialog,all_length,maxlen,init_hidden,w_index,w_batch_index,all_words
        else:
            return tokenized_dialog,all_length,maxlen,init_hidden,w_index,w_batch_index
        
