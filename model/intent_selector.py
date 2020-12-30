import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import segment_coo


class IntentSelector(nn.Module):
    def __init__(self,utter_embed_size,atten_hidden=20):
        super(IntentSelector,self).__init__()
        self.utter_embed_size=utter_embed_size
        self.atten_hidden=20
        # self.chat_att=Attention(utter_embed_size,graph_embed_size,atten_hidden)
        # self.question_att=Attention(utter_embed_size,graph_embed_size,atten_hidden)
        # self.recommend_att=Attention(utter_embed_size,graph_embed_size,atten_hidden)
        # self.Wf=nn.Linear(3*graph_embed_size,3)
        self.W1=nn.Linear(utter_embed_size,64)
        self.W2=nn.Linear(64,3)


    def forward(self,utterance_embed):#,graph_embedding,mention_history,candidate_index,general_index,graph_size):
        #last_utterance=utterance_embed[:,-1,:]
        #print(last_utterance)
        layer1=torch.relu(self.W1(utterance_embed))
        intents=self.W2(layer1)
        return intents
        