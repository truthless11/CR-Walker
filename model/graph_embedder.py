import torch
import torch.nn as nn
import json
import math
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv,GCNConv
import os.path as osp

root=osp.dirname(osp.dirname(osp.abspath(__file__)))

class Graph_Embedder(nn.Module):
    def __init__(self,num_nodes=30471,embed_size=64,num_turns=10,num_relations=12,num_bases=30,device_str='cuda:1',word_net=False):
        super(Graph_Embedder,self).__init__()
        self.num_turns=num_turns
        self.embed_size=embed_size
        self.num_nodes=num_nodes
        self.word_net=word_net
        self.device=torch.device(device_str)
        self.bow_size=512
        self.gcn1 = RGCNConv(self.embed_size,self.embed_size,num_relations,num_bases)

        self.num_words=29308
        if word_net:
            self.gcn2 = GCNConv(self.embed_size, self.embed_size)
            self.concept_edge_sets = self.concept_edge_list4GCN()
        #self.gcn2 = RGCNConv(self.embed_size,self.embed_size,num_relations,num_bases)
        self.initialize_weights()
        #self.init_features=nn.Parameter(,device=self.device))
        #self.mention_W = nn.Linear(2*rnn_hidden,1)


    def forward(self,edge_type,edge_index):
        graph_features=torch.relu(self.gcn1(self.init_features,edge_index,edge_type))


        if self.word_net:
            word_features=self.gcn2(self.init_word_features.weight,self.concept_edge_sets)
            return graph_features, word_features
        else:
            return graph_features, None

    def initialize_weights(self):
        features=torch.Tensor(self.num_nodes,self.embed_size).uniform_(-1/math.sqrt(self.embed_size),1/math.sqrt(self.embed_size))
        features_norm=torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12, out=None)
        self.init_features=nn.Parameter(features_norm)


        if self.word_net:
            self.init_word_features = nn.Embedding(self.num_words, self.embed_size)
            nn.init.normal_(self.init_word_features.weight, mean=0, std=self.embed_size ** -0.5)
            nn.init.constant_(self.init_word_features.weight[0], 0)

    def concept_edge_list4GCN(self):
        node2index=json.load(open(osp.join(root,"data","key2index_3rd.json"),encoding='utf-8'))
        f=open(osp.join(root,"data","conceptnet_edges2nd.txt"),encoding='utf-8')
        edges=set()
        stopwords=set([word.strip() for word in open(osp.join(root,"data","stopwords.txt"),encoding='utf-8')])
        for line in f:
            lines=line.strip().split('\t')
            entity0=node2index[lines[1].split('/')[0]]
            entity1=node2index[lines[2].split('/')[0]]
            if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
                continue
            edges.add((entity0,entity1))
            edges.add((entity1,entity0))
        edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
        return torch.LongTensor(edge_set).cuda()