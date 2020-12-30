import torch
import torch.nn as nn
import math
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

class Graph_Embedder(nn.Module):
    def __init__(self,num_nodes=30471,embed_size=64,num_turns=10,num_relations=12,num_bases=30,device_str='cuda:1',):
        super(Graph_Embedder,self).__init__()   
        self.num_turns=num_turns
        self.embed_size=embed_size
        self.num_nodes=num_nodes
        self.device=torch.device(device_str)
        self.gcn1 = RGCNConv(self.embed_size,self.embed_size,num_relations,num_bases)
        #self.gcn2 = RGCNConv(self.embed_size,self.embed_size,num_relations,num_bases)
        self.initialize_weights()
        #self.init_features=nn.Parameter(,device=self.device))
        #self.mention_W = nn.Linear(2*rnn_hidden,1)


    def forward(self,edge_type,edge_index):
        #utterance_embed:(batch_size,num_turns,rnn_hidden)
        #node_features:(num_nodes,9)
        #self.batch_size=utterance_embed.size()[0]
        #num_node=node_features.size()[0]

        #print(self.init_features.size())
        #print(edge_index.size())
        #print(edge_type.size())
        graph_features=torch.relu(self.gcn1(self.init_features,edge_index,edge_type))
        #print(hid_features.size())
        #graph_features=self.gcn2(hid_features,edge_index,edge_type)

        #graph_embed=torch.cat((hid_features,node_features),dim=-1)

        #graph_embed=torch.cat((node_features,mention_features),dim=1)
        # pad=torch.zeros_like(graph_embed[0]).unsqueeze(0)
        # graph_embed=torch.cat((graph_embed,pad),dim=0)
        return graph_features#self.init_features
        #return self.init_features
    
    def initialize_weights(self):
        features=torch.Tensor(self.num_nodes,self.embed_size).uniform_(-1/math.sqrt(self.embed_size),1/math.sqrt(self.embed_size))
        features_norm=torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12, out=None)
        self.init_features=nn.Parameter(features_norm)


    # def get_mention_features(self,utterance_embed,mention_indice,init_features,graph_size):
    #     utterance_embed=utterance_embed.permute(1,0,2)
    #     mention_features=init_features
    #     #init_features(num_nodes,embed_size)      
    #     for k in range(self.num_turns):
    #         turn_utter=utterance_embed[k]#(batch_size,rnn_hidden)
    #         tiled_turn=self.tile_utterance(turn_utter,graph_size)#(num_nodes,rnn_hidden)
    #         concat=torch.cat((mention_features,tiled_turn),dim=1)#(num_nodes,embed_size+rnn_hidden)
    #         lambda_=torch.sigmoid(self.mention_W(concat))#(num_nodes,1)
    #         updated=mention_features.mul(lambda_)+tiled_turn.mul(1-lambda_)
    #         mention_features=mention_features.masked_scatter(mention_indice[k],updated)
    #     return mention_features


    # def tile_utterance(self,turn_utter,graph_size):
    #     #turn_utter(batch_size,rnn_hidden)
    #     #graph_size(batch_size)
    #     tile_utter=[]
    #     for i in range(self.batch_size):
    #         repeated=turn_utter[i].repeat(graph_size[i],1)
    #         tile_utter.append(repeated)
    #     tile_utter=torch.cat(tile_utter,dim=0)
    #     return tile_utter

    # def get_mention_indice(self,mention_history,graph_size,num_node):
    #     mention_indice=torch.zeros(self.num_turns,num_node,1)
    #     offset=0
    #     for i in range(len(mention_history)):
    #         turns=len(mention_history[i])
    #         if turns>self.num_turns:
    #             processed=mention_history[i][turns-self.num_turns:]
    #         elif turns<self.num_turns:
    #             pad=[]
    #             for j in range(self.num_turns-turns):
    #                 pad.append([])
    #             processed=pad+mention_history[i]
    #         else:
    #             processed=mention_history[i]

    #         for j in range(self.num_turns):
    #             for p in processed[j]:
    #                 mention_indice[j][p+offset]=1
    #         offset+=graph_size[i]
    #     return mention_indice.bool()
    

