import sys
from transformers import BertModel,BertTokenizer
from graph_embedder import Graph_Embedder



import torch
import torch.nn as nn
import os.path as osp
##from transformers import BertModel,BertTokenizer

from torch_geometric.data import DataLoader




root=osp.dirname(osp.dirname(osp.abspath(__file__)))
path=osp.join(root,"data","gorecdial")





# class Utterance_Embedder(nn.Module):
#     def __init__(self,rnn_type="RNN_TANH",use_bert=True,rnn_hidden=64,dropout=0.5,maxlen=128,num_turns=10):
#         super(Utterance_Embedder,self).__init__()   
#         model_class=BertModel 
#         pretrained_weights = 'bert-base-uncased'
#         self.rnn_hidden=rnn_hidden
#         self.maxlen=maxlen
#         self.num_turns=num_turns
#         self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
#         self.model = model_class.from_pretrained(pretrained_weights)
#         self.device = torch.device('cuda:1')

#         if rnn_type in ['LSTM', 'GRU']:
#             self.rnn = getattr(nn, rnn_type)(768, rnn_hidden,dropout=dropout)
#         else:
#             try:
#                 nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
#             except KeyError:
#                 raise ValueError( """An invalid option for `--model` was supplied,
#                                  options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
#             self.rnn = nn.RNN(768, rnn_hidden, nonlinearity=nonlinearity, dropout=dropout)
        
#         self.rnn_type=rnn_type

#     def forward(self,dialog_history):
#         #dialog_history(batch_size,num_turns)
#         utterance_embed=[]
#         dialog_history=self.pad_dialog(dialog_history)
#         #print(dialog_history)
#         for dialog in dialog_history:
#             hidden=self.get_init_hidden()
#             batch_turns=[]
#             for turn in dialog:
#                 if turn=="":
#                     batch_turns.append(torch.zeros(self.rnn_hidden,device=self.device))
#                     continue
#                 input_ids = torch.tensor([self.tokenizer.encode(turn)],device=self.device)
#                 bert_embed = self.model(input_ids)[0]
#                 bert_embed = bert_embed.permute(1,0,2)
#                 output,hidden=self.rnn(bert_embed,hidden)
#                 if self.rnn_type=="LSTM":
#                     batch_turns.append(hidden[-1].squeeze())
#                 else:
#                     batch_turns.append(hidden.squeeze())
#             utterance_embed.append(torch.stack(batch_turns))
#         utterance_embed=torch.stack(utterance_embed)
#         #print(utterance_embed.size())
#         return utterance_embed
    
#     def get_init_hidden(self):
#         if self.rnn_type=="LSTM":
#             hidden=(torch.zeros(1,1,self.rnn_hidden),torch.zeros(1,1,self.rnn_hidden)).cuda(device=torch.device('cuda:1'))
#         else:
#             hidden=torch.zeros(1,1,self.rnn_hidden).cuda(device=self.device)
#         return hidden
#     def pad_dialog(self,dialog_history):
#         pad_history=[]
#         for i in range(len(dialog_history)):
#             turns=len(dialog_history[i])
#             if turns>self.num_turns:
#                 padded=dialog_history[i][turns-self.num_turns:]
#                 pad_history.append(padded)
#             elif turns<self.num_turns:
#                 pad=[]
#                 for j in range(self.num_turns-turns):
#                     pad.append("")
#                 padded=pad+dialog_history[i]
#                 pad_history.append(padded)
#             else:
#                 pad_history.append(dialog_history[i])
#         return pad_history
        

class Utterance_Embedder(nn.Module):
    def __init__(self,rnn_type="RNN_TANH",use_bert=True,rnn_hidden=64,dropout=0.5,num_turns=10):
        super(Utterance_Embedder,self).__init__()   
        model_class=BertModel 
        pretrained_weights = 'bert-base-uncased'
        self.rnn_hidden=rnn_hidden
        #self.maxlen=maxlen
        self.num_turns=num_turns
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

    def forward(self,tokenized,length,max_len,init_hidden):
        #dialog_history(batch_size,num_turns)
        #print(dialog_history)
        #batch_size=len(dialog_history)
        #utterance_embed=[]
        #print("start!")
        bert_embed=self.model(**tokenized)[0].view(-1,self.num_turns,max_len,768).permute(1,0,2,3)#(num_turns,batch_size,max_len,768)
        #print("end!")
        sentence_rep=bert_embed[:,:,0,:]
        
        #print(dialog_history)
        turn_hidden=[]
        hidden=init_hidden

        output,hidden=self.rnn(sentence_rep,hidden)
        # for i in range(self.num_turns):
        #     turn_embed=bert_embed[i]
        #     turn_length=length[i]
        #     packed_embed = nn.utils.rnn.pack_padded_sequence(input=turn_embed, lengths=turn_length, batch_first=True,enforce_sorted=False)
        #     output,hidden=self.rnn(packed_embed,hidden)
        #     if self.rnn_type=="LSTM":
        #         turn_hidden.append(hidden[-1].squeeze())
        #     else:
        #         turn_hidden.append(hidden.squeeze())
        output=output[-1,:,:]
        
        # for dialog in dialog_history:
        #     hidden=self.get_init_hidden()
        #     batch_turns=[]
        #     for turn in dialog:
        #         if turn=="":
        #             batch_turns.append(torch.zeros(self.rnn_hidden))
        #             continue
        #         input_ids = torch.tensor([self.tokenizer.encode(turn)])
                
        #         bert_embed = bert_embed.permute(1,0,2)
        #         output,hidden=self.rnn(bert_embed,hidden)
        #        
        #utterance_embed=torch.stack(turn_hidden).permute(1,0,2)
        # utterance_embed=torch.stack(utterance_embed)
        #print(utterance_embed.size())
        return output
    
    def get_init_hidden(self,batch_size):
       
        return hidden


    def prepare_data(self,dialog_history,device):
        pad_history=[]
        for i in range(len(dialog_history)):
            turns=len(dialog_history[i])
            if turns>self.num_turns:
                padded=dialog_history[i][turns-self.num_turns:]
                pad_history=pad_history+padded
            elif turns<self.num_turns:
                pad=[]
                for j in range(self.num_turns-turns):
                    pad.append("")
                padded=pad+dialog_history[i]
                pad_history=pad_history+padded
            else:
                pad_history=pad_history+dialog_history[i]
        #print(len(pad_history))
        tokenized_dialog=self.tokenizer.batch_encode_plus(pad_history,pad_to_max_length=True,return_tensors="pt",)#[turn1,turn2,turn3....]
        #print(tokenized_dialog)
        for key in tokenized_dialog.keys():
            tokenized_dialog[key]=tokenized_dialog[key].to(device=device)
        all_length=torch.sum(tokenized_dialog["attention_mask"],dim=-1).view(-1,self.num_turns).permute(1,0)
        maxlen=tokenized_dialog["attention_mask"].size()[-1]
        batch_size=len(dialog_history)

        if self.rnn_type=="LSTM":
            init_hidden=(torch.zeros(1,batch_size,self.rnn_hidden),torch.zeros(batch_size,1,self.rnn_hidden)).to(device=device)
        else:
            init_hidden=torch.zeros(1,batch_size,self.rnn_hidden).to(device=device)

        return tokenized_dialog,all_length,maxlen,init_hidden
        


# class Utterance_Embedder(nn.Module):
#     def __init__(self,rnn_type="RNN_TANH",use_bert=True,rnn_hidden=128,dropout=0.5,num_turns=10):
#         super(Utterance_Embedder,self).__init__()   
#         model_class=BertModel 
#         pretrained_weights = 'bert-base-uncased'
#         #self.rnn_hidden=rnn_hidden
#         #self.maxlen=maxlen
#         self.num_turns=num_turns
#         self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
#         self.model = model_class.from_pretrained(pretrained_weights)
#         self.Wu=nn.Linear(768,rnn_hidden)
#         #self.device = torch.device('cuda:1')
#         # if rnn_type in ['LSTM', 'GRU']:
#         #     self.rnn = getattr(nn, rnn_type)(768, rnn_hidden,dropout=dropout)
#         # else:
#         #     try:
#         #         nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
#         #     except KeyError:
#         #         raise ValueError( """An invalid option for `--model` was supplied,
#         #                          options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
#         #     self.rnn = nn.RNN(768, rnn_hidden, nonlinearity=nonlinearity, dropout=dropout)
        
#         #self.rnn_type=rnn_type

#     def forward(self,tokenized,pretrain=False):
#         #dialog_history(batch_size,num_turns)
#         #print(dialog_history)
#         #batch_size=len(dialog_history)
#         #utterance_embed=[]
#         #print("start!")
#         bert_embed=self.model(**tokenized)[0]#.view(-1,self.num_turns,max_len,768).permute(1,0,2,3)#(num_turns,batch_size,max_len,768)
#         #print("end!")
#         sentence_rep=bert_embed[:,0,:]

#         output=self.Wu(sentence_rep)
#         #print(output.size())


        
#         #print(dialog_history)
#         #turn_hidden=[]
#         #hidden=init_hidden

#         #output,hidden=self.rnn(sentence_rep,hidden)
#         # for i in range(self.num_turns):
#         #     turn_embed=bert_embed[i]
#         #     turn_length=length[i]
#         #     packed_embed = nn.utils.rnn.pack_padded_sequence(input=turn_embed, lengths=turn_length, batch_first=True,enforce_sorted=False)
#         #     output,hidden=self.rnn(packed_embed,hidden)
#         #     if self.rnn_type=="LSTM":
#         #         turn_hidden.append(hidden[-1].squeeze())
#         #     else:
#         #         turn_hidden.append(hidden.squeeze())
#        # output=output.permute(1,0,2)
        
#         # for dialog in dialog_history:
#         #     hidden=self.get_init_hidden()
#         #     batch_turns=[]
#         #     for turn in dialog:
#         #         if turn=="":
#         #             batch_turns.append(torch.zeros(self.rnn_hidden))
#         #             continue
#         #         input_ids = torch.tensor([self.tokenizer.encode(turn)])
                
#         #         bert_embed = bert_embed.permute(1,0,2)
#         #         output,hidden=self.rnn(bert_embed,hidden)
#         #        
#         #utterance_embed=torch.stack(turn_hidden).permute(1,0,2)
#         # utterance_embed=torch.stack(utterance_embed)
#         if pretrain==True:
#             return output,sentence_rep
#         #print(utterance_embed.size())
#         else:
#             return output
    
#     # def get_init_hidden(self,batch_size):
       
#     #     return hidden


#     def prepare_data(self,dialog_history,device):
#         pad_history=[]
#         # for i in range(len(dialog_history)):
#         #     turns=len(dialog_history[i])
#         #     if turns>self.num_turns:
#         #         padded=dialog_history[i][turns-self.num_turns:]
#         #         pad_history=pad_history+padded
#         #     elif turns<self.num_turns:
#         #         pad=[]
#         #         for j in range(self.num_turns-turns):
#         #             pad.append("")
#         #         padded=pad+dialog_history[i]
#         #         pad_history=pad_history+padded
#         #     else:
#         #         pad_history=pad_history+dialog_history[i]

#         for bat in dialog_history:
#             history=""
#             turns=len(bat)
#             padded=bat
#             if turns>self.num_turns:
#                 padded=bat[turns-self.num_turns:]
#                 #pad_history=pad_history+padded
#             for sentence in padded:
#                 history+=sentence
#             pad_history.append(history)
#         #print(len(pad_history))
#         tokenized_dialog=self.tokenizer.batch_encode_plus(pad_history,max_length=512,pad_to_max_length=True,truncation=True,return_tensors="pt",)#[turn1,turn2,turn3....]
#         #print(tokenized_dialog)
#         for key in tokenized_dialog.keys():
#             tokenized_dialog[key]=tokenized_dialog[key].to(device=device)
#         #all_length=torch.sum(tokenized_dialog["attention_mask"],dim=-1).view(-1,self.num_turns).permute(1,0)
#         #maxlen=tokenized_dialog["attention_mask"].size()[-1]
#         #batch_size=len(dialog_history)

#         # if self.rnn_type=="LSTM":
#         #     init_hidden=(torch.zeros(1,batch_size,self.rnn_hidden),torch.zeros(batch_size,1,self.rnn_hidden)).to(device=device)
#         # else:
#         #     init_hidden=torch.zeros(1,batch_size,self.rnn_hidden).to(device=device)

#         return tokenized_dialog#,all_length,maxlen#,init_hidden
        

if __name__ == "__main__":
    #utter_embedder=Utterance_Embedder()
    gorecdial=GoRecDial(path,flag="train")
    data=GoRecDial(path,flag="bow")[0]
    # #loader=DataLoader(gorecdial,batch_size=5,shuffle=True)
    # print(data.bows[0])
    
    loader = torch.utils.data.DataLoader(gorecdial,batch_size=5)
    #utter_embed=Utterance_Embedder()
    
    #device = torch.device('cuda:1')
    #utter_embed.to(device)
    #graph_embed=Graph_Embedder()
    # all_intent=['chat','question','recommend']
    # distrib=[0,0,0]
    for q,batch in enumerate(loader):
        print(batch.node_candidate1)


        # mention_index=[]
        # mention_batch_index=[]

        # for i,batch in enumerate(batch.mention_history):
        #     mentioned=[]
        #     for item in batch:
        #         for k in item:
        #             if k not in mentioned:
        #                 mention_index.append(k)
        #                 mention_batch_index.append(i)
        #                 mentioned.append(k)
        # mention_index=torch.Tensor(mention_index).long()
        # mention_batch_index=torch.Tensor(mention_batch_index).long()

        #print(mention_index)
        #print(mention_batch_index)
        

        # print(batch.node_candidate2)
        # print(batch.oracle_node2)
        # sel_indices=[]
        # sel_batch_indices=[]


        # bat=[[],[],[]]
        # bad=[[],[],[]]
        # good=[[],[],[]]


        # good_index_1=[]
        # bad_index_1=[]
        # sel_batch_index_1=[]

        # for i,item in enumerate(batch.node_candidate1):
        #     oracles=batch.oracle_node1[i]
        #     for j,oracle in enumerate(oracles):
        #         for idx,k in enumerate(item):
        #             if idx!=oracle:
        #                 bat[j].append(i)
        #                 bad[j].append(k)
        #                 good[j].append(item[oracle])



        # for item in bad:
        #     bad_index_1.append(torch.Tensor(item).long())
        
        # for item in good:
        #     good_index_1.append(torch.Tensor(item).long())

        # for item in bat:
        #     sel_batch_index_1.append(torch.Tensor(item).long())
        

        # #print(bad_index_1)
        # #print(good_index_1)
        # #print(sel_batch_index_1)
        # # sel_index=torch.Tensor(sel).long()
        # # batch_index=torch.Tensor(bat).long()


        


        # good_index_2=[]
        # bad_index_2=[]
        # sel_batch_index_2=[]


        # bat=[[],[],[]]
        # bad=[[],[],[]]
        # good=[[],[],[]]
        # # bat=[[],[],[]]
        # # sel=[[],[],[]]
        # for i,bth in enumerate(batch.node_candidate2):#batch
        #     oracles=batch.oracle_node2[i]
        #     for j,item in enumerate(bth):#path_num
        #         for idx,k in enumerate(item):
        #             if idx!=oracles[j]:
        #                 bat[j].append(i)
        #                 bad[j].append(k)
        #                 good[j].append(item[oracles[j]])

        
        # for item in bad:
        #     bad_index_2.append(torch.Tensor(item).long())
        
        # for item in good:
        #     good_index_2.append(torch.Tensor(item).long())

        # for item in bat:
        #     sel_batch_index_2.append(torch.Tensor(item).long())

        # print(bad_index_2)
        # print(good_index_2)
        # print(sel_batch_index_2)
        
        # for item in sel:
        #     sel_index_2.append(torch.Tensor(item).long())
        # for item in bat:
        #     sel_batch_index_2.append(torch.Tensor(item).long())


        # sel_indices=[sel_index_1,sel_index_2]
        # sel_batch_indices=[sel_batch_index_1,sel_batch_index_2]

        # #print(sel_index)
        #print(batch_index)
        #break
        # intent=batch.intent
        # for item in intent:
        #     distrib[all_intent.index(item)]+=1
        # if q==50:
        #     break

    #     # print(batch.dialog_history)
    #     # print(batch.mention_history)
    #print(distrib)
        #a=utter_embed.forward(batch.dialog_history)
    #graph_embeddings=graph_embed.forward(graph.node_feature,graph.edge_type,graph.edge_index)
        #print(a.size())
    # for batch in loader:
    #     #print(batch)
    #     print(batch.mention_history)
    #     #print(batch.graph_size)
    #     embeddings=utter_embed.forward(batch.dialog_history)
    #     print("utterance_embedding:",embeddings.size())

        #graph_embeddings=graph_embed.forward(embeddings,batch.node_feature,batch.edge_type,batch.edge_index,batch.mention_history,batch.graph_size)
        #print("graph_embedding:",graph_embeddings.size()
    #utter_embedder.forward([["","how was your day?","I'm feeling good, and you?"],["what's your favorite movie?","I like horror and comics.","Oh, I see."]])
