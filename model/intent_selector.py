import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import segment_coo


class IntentSelector(nn.Module):
    def __init__(self,utter_embed_size,atten_hidden=20):
        super(IntentSelector,self).__init__()
        self.utter_embed_size=utter_embed_size
        self.atten_hidden=20
        self.W1=nn.Linear(utter_embed_size,64)
        self.W2=nn.Linear(64,3)


    def forward(self,utterance_embed):
        layer1=torch.relu(self.W1(utterance_embed))
        intents=self.W2(layer1)
        return intents
        