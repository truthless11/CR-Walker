from torch_geometric.data import InMemoryDataset, Dataset, download_url, Data
import os.path as osp
import numpy as np
import json
import random
import torch
import copy
from tqdm import tqdm



class GoRecDial(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,flag="train"):
        super(GoRecDial, self).__init__(root, transform, pre_transform)
        self.flag=flag
        if self.flag=="test":
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.flag=="graph":
            self.data, self.slices = torch.load(self.processed_paths[2])
        elif self.flag=="bow":
            self.data, self.slices = torch.load(self.processed_paths[3])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

        self.train_size=8213
        self.test_size=909

    @property
    def raw_file_names(self):
        return ['test.json','train.json','gorecdial_kg.json','global_bow.json']

    @property
    def processed_file_names(self):
        return ['train.pt','test.pt','graph.pt','bow.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):

        test_reason_path=self.raw_paths[0]
        train_reason_path=self.raw_paths[1]
        graph_path=self.raw_paths[2]
        bow_path=self.raw_paths[3]
        f=open(test_reason_path)
        test_reason_path=json.load(f) 
        f=open(train_reason_path)
        train_reason_path=json.load(f)
        f=open(graph_path)
        graph=json.load(f)
        f=open(bow_path)
        bow=json.load(f)
        flags=['train','test']

        relations_names=['time', 'director', 'starring', 'genre', 'subject', 'belong', 'timeR', 'directorR', 'starringR', 'genreR', 'subjectR', 'belongR']
        type_names = ["Candidate","Movie","Actor","Director","Genre","Time","Attr","Subject","None"]

        for q,flag in enumerate(flags):
            data_list = []
            tot=0
            if flag=='train':
                reason_path=train_reason_path
            else:
                reason_path=test_reason_path
            organized_data=[]


            for idx in tqdm(range(len(reason_path))):
                last_turn=0
                if idx==len(reason_path)-1:
                    last_turn=1
                else:
                    if reason_path[idx+1]['dialog_num']!=reason_path[idx]['dialog_num']:
                        last_turn=1
                key=str(reason_path[idx]['dialog_num'])+"_"+str(reason_path[idx]['system_turn'])

                data=Data(dialog_history=reason_path[idx]['context'],oracle_response=reason_path[idx]['utterance'],mention_history=reason_path[idx]['mentioned'],intent=reason_path[idx]['intent'],node_candidate1=reason_path[idx]['node_candidate1'],label_1=reason_path[idx]['label_1'],label_c=reason_path[idx]['label_c'],node_candidate2=reason_path[idx]['node_candidate2'],label_2=reason_path[idx]['label_2'],rec_cand=reason_path[idx]['rec_cand'],new_mention=reason_path[idx]['new_mentioned'],my_id=key,last_turn=last_turn)


                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[q])

        nodes=graph['nodes']
        relations=graph['relations']
        

        
        edge_index=[[],[]]
        edge_type=[]
        
        num_nodes=len(nodes)
        node_feature=torch.zeros(num_nodes,9)
        for i,node in enumerate(nodes):
            if node['type']=="Person":
                for item in node['role']:
                    type_idx=type_names.index(item)
                    node_feature[i][type_idx]=1
            else:
                type_idx=type_names.index(node['type'])
                node_feature[i][type_idx]=1

        

        

        for relation in relations:
            edge_index[0].append(int(relation[0]))
            edge_index[1].append(int(relation[1]))
            edge_type.append(relations_names.index(relation[2]))
        
        edge_index=torch.from_numpy(np.array(edge_index))
        edge_type=torch.from_numpy(np.array(edge_type)).long()

        data=Data(edge_index=edge_index,edge_type=edge_type,num_nodes=num_nodes,graph_size=num_nodes,node_feature=node_feature)
        data_list=[data]
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[2])


        all_bow=[]
        for i,node in enumerate(nodes):
            if node['type']=="Movie":
                if str(i) in bow.keys():
                    all_bow.append(torch.Tensor(bow[str(i)]))
                else:
                    all_bow.append(torch.zeros(512))
            else:
                break
        
        all_bow=torch.stack(all_bow)
        data=Data(bow_embed=all_bow)
        data_list=[data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[3])

if __name__ == "__main__":
    root=osp.dirname(osp.dirname(osp.abspath(__file__)))
    path=osp.join(root,"data","gorecdial")
    dataset=GoRecDial(path,flag="test")
    data=dataset[10]
    print(data.intent)
    print(data.node_candidate1)
    print(data.label_1)
    print(data.node_candidate2)
    print(data.label_2)