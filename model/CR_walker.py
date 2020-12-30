
import random
import torch
import torch.nn as nn
import copy
from torch_geometric.data import Data
from utterance_embedder import Utterance_Embedder
from graph_embedder import Graph_Embedder
from intent_selector import IntentSelector
from graph_walker import Graph_Walker
from explicit_recommender import Explicit_Recommender

class ProRec(nn.Module):
    #beta=tanh(QW+KU)*V
    def __init__(self,device_str='cuda:1',rnn_type="RNN_TANH",use_bert=True,utter_embed_size=64,dropout=0.5,num_turns=10,num_relations=12,num_bases=15,graph_embed_size=64,atten_hidden=20,negative_sample_ratio=3,dataset="redial"):
        super(ProRec,self).__init__()
        self.dataset=dataset
        if dataset=="redial":
            self.num_nodes=30471
        else:
            self.num_nodes=19308
        self.use_bert=use_bert
        self.utter_embed_size=utter_embed_size
        self.dropout=dropout
        self._lambda1=0.1
        self._lambda2=0.01
        #self.maxlen=maxlen
        self.num_turns=num_turns
        self.num_relations=num_relations
        self.num_bases=num_bases
        self.graph_embed_size=graph_embed_size
        self.atten_hidden=atten_hidden
        

        self.device=torch.device(device_str)
        self.negative_sample_ratio=negative_sample_ratio

        self.utter_embedder=Utterance_Embedder(rnn_type,use_bert,utter_embed_size,dropout,num_turns)
        self.graph_embedder=Graph_Embedder(num_nodes=self.num_nodes,embed_size=graph_embed_size,device_str=device_str)
        self.intent_selector=IntentSelector(utter_embed_size,atten_hidden)
        self.graph_walker=Graph_Walker(attention_hidden_dim=atten_hidden,graph_embed_size=graph_embed_size,utterance_embed_size=utter_embed_size,device_str=device_str,nagetive_sample_ratio=negative_sample_ratio)
        if self.dataset=="gorecdial":
            self.explicit_recommender=Explicit_Recommender(utterance_embed_size=utter_embed_size,graph_embed_size=graph_embed_size)#utterance_embed_size=utter_embed_size,graph_embed_size=graph_embed_size)


        self.walk_loss_1=torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.walk_loss_2=nn.BCEWithLogitsLoss(reduction="sum")
        self.intent_loss=torch.nn.CrossEntropyLoss(reduction='sum')
        self.rec_loss=torch.nn.CrossEntropyLoss(reduction='sum')

        self.alignment_loss=nn.BCEWithLogitsLoss(reduction="sum")
        #self.alignment_loss=nn.MSELoss()
        self.Wa=nn.Linear(utter_embed_size,graph_embed_size,bias=False)
        #self.Wo=nn.Linear(1,self.num_nodes)



    def forward_pretrain(self,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #last_utter=utter_embed[:,-1,:]
        #print(last_utter)
        intent=self.intent_selector.forward(utter_embed)
        #print(alignment_index)
        graph_embed=self.graph_embedder.forward(edge_type,edge_index)
        graph_features=graph_embed.index_select(0,alignment_index)

        #print(graph_features)
        tiled_utter=self.graph_walker.tile_context(utter_embed,alignment_batch_index)
        logits=torch.sum(self.Wa(tiled_utter)*graph_features,dim=-1)
        #print(logits)
        #logits=logits+self.Wo.bias.index_select(0,alignment_index)
        
        loss_a=self.alignment_loss(logits,alignment_label)
        intent_loss=self.intent_loss(intent,intent_label)
        return loss_a+intent_loss
    
    def prepare_pretrain(self,mention_history,intent=None,dialog_history=None,edge_type=None,edge_index=None):
        alignment_index=[]
        alignment_batch_index=[]
        alignment_label=[]
        for i,item in enumerate(mention_history):
            mentioned=[]
            for k in item:
                alignment_index.append(k)
                alignment_batch_index.append(i)
                alignment_label.append(1)
                mentioned.append(k)
            len_positive=len(mentioned)
            if len(mentioned)==0:
                len_positive+=1
                mentioned.append(30458)
                alignment_index.append(30458)
                alignment_batch_index.append(i)
                alignment_label.append(1)
            else:
                alignment_index.append(30458)
                alignment_batch_index.append(i)
                alignment_label.append(0)

            
            sampled=[]
            for ite in range(4*len_positive):
                #cand=0
                cand=random.sample(range(self.num_nodes),1)[0]
                while cand in mentioned or cand in sampled:
                    cand=random.sample(range(self.num_nodes),1)[0]
                sampled.append(cand)
                alignment_index.append(cand)
                alignment_batch_index.append(i)
                alignment_label.append(0)
            
        alignment_index=torch.Tensor(alignment_index).long().to(device=self.device)
        alignment_batch_index=torch.Tensor(alignment_batch_index).long().to(device=self.device)
        alignment_label=torch.FloatTensor(alignment_label).to(device=self.device)

        

        if dialog_history!=None:
            tokenized_dialog,all_length,maxlen,init_hidden = self.utter_embedder.prepare_data(dialog_history,self.device)
            edge_index=edge_index.to(device=self.device)
            edge_type=edge_type.to(device=self.device)
            all_intent=["chat","question","recommend"]
            batch_size=len(intent)
            intent_label=torch.zeros(batch_size,device=self.device)
            for i in range(batch_size):
                intent_label[i]=all_intent.index(intent[i])
            intent_label=intent_label.long()
            return tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label
        else:
            return alignment_index,alignment_batch_index,alignment_label

    
    def forward(self,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,alignment_index,alignment_batch_index,alignment_label):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed=self.graph_embedder.forward(edge_type,edge_index)
        #print("graph_embedding:",graph_embed.size())
        intent=self.intent_selector.forward(utter_embed)#,graph_embeddings,mention_history,self.get_group_index(cur_type,"Candidate"),self.get_group_index(cur_type,"Attr"),graph_size)
        #print("intent:",intent.size())
        paths=self.graph_walker.forward(graph_embed,utter_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices)
        walk_loss_1=self.walk_loss_1(paths[0],label_1)
        walk_loss_2=self.walk_loss_2(paths[1],label_2)
        intent_loss=self.intent_loss(intent,intent_label)

        graph_features=graph_embed.index_select(0,alignment_index)
        tiled_utter=self.graph_walker.tile_context(utter_embed,alignment_batch_index)
        logits=torch.sum(self.Wa(tiled_utter)*graph_features,dim=-1)
        loss_a=self.alignment_loss(logits,alignment_label)


        tot_loss=walk_loss_1+intent_loss+0.025*loss_a#+walk_loss_2
        return intent,paths,tot_loss

    
    def forward_gorecdial(self,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed=self.graph_embedder.forward(edge_type,edge_index)
        intent=self.intent_selector.forward(utter_embed)
        #print("graph_embedding:",graph_embed.size())
        #,graph_embeddings,mention_history,self.get_group_index(cur_type,"Candidate"),self.get_group_index(cur_type,"Attr"),graph_size)
        #print("intent:",intent.size())
        #paths=self.graph_walker.forward(graph_embed,utter_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,intent_indices,node_feature)
        paths,user_portrait=self.graph_walker.forward(graph_embed,utter_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,ret_portrait=True)

        

        rec=self.explicit_recommender.forward(utter_embed,graph_embed,user_portrait,bow_embed,rec_index,rec_batch_index)

        #for i,item in enumerate(paths):
            #print("hop "+str(i)+":",item.size())
        #loss1=self.walk_loss.forward(graph_embed,paths,bad_indices,batch_indices,golden_indices)
        walk_loss_1=self.walk_loss_1(paths[0],label_1)
        walk_loss_2=self.walk_loss_2(paths[1],label_2)
        intent_loss=self.intent_loss(intent,intent_label)
        rec_loss=self.rec_loss(rec,rec_golden)
        #print(rec_golden.unsqueeze(-1))
        #one_hot_golden = torch.zeros(10, 5).cuda(device=self.device).scatter_(1, rec_golden.unsqueeze(-1), 1)
        #print(one_hot_golden)
        

        tot_loss=self._lambda1*walk_loss_1+self._lambda2*walk_loss_2+intent_loss+rec_loss
        return intent,paths,tot_loss



    def get_intent(self,tokenized_dialog,all_length,maxlen,init_hidden):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        intent=self.intent_selector.forward(utter_embed)
        return intent
        

    def inference_gorecdial(self,intent,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,bow_embed=None,layer=0):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed=self.graph_embedder.forward(edge_type,edge_index)

        user_portrait=self.graph_walker.get_user_portrait(mention_index,mention_batch_index,graph_embed)

        step=self.graph_walker.forward_single_layer(layer,utter_embed,user_portrait,graph_embed,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,intent)
        if bow_embed!=None:
            rec=self.explicit_recommender.forward(utter_embed,graph_embed,user_portrait,bow_embed,sel_index,sel_batch_index)
            return step,rec
        else:
            return step
    
    def inference_redial(self,intent,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,layer=0):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed=self.graph_embedder.forward(edge_type,edge_index)

        user_portrait=self.graph_walker.get_user_portrait(mention_index,mention_batch_index,graph_embed)

        step=self.graph_walker.forward_single_layer(layer,utter_embed,user_portrait,graph_embed,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,intent)
        return step
    
    # def get_rec(self,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,node_feature,bow_embed,rec_index,rec_batch_index):
    #     utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
    #     #print("utterance_embedding:",utter_embed.size())
    #     graph_embed=self.graph_embedder.forward(edge_type,edge_index)

    #     rec=self.explicit_recommender.forward(utter_embed,graph_embed,bow_embed,rec_index,rec_batch_index)
    #     return rec
    def prepare_data_redial(self,dialog_history,mention_history,intent,node_candidate1,node_candidate2,edge_type,edge_index,label1,label2,sample=False):
        #tokenized_dialog,all_length,maxlen,init_hidden = self.utter_embedder.prepare_data(dialog_history,self.device)
        tokenized_dialog,all_length,maxlen,init_hidden = self.utter_embedder.prepare_data(dialog_history,self.device)
        mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,label_1,label_2=self.graph_walker.prepare_data(mention_history,intent,node_candidate1,node_candidate2,label1,label2,self.device,sample=sample,dataset="redial")
        edge_index=edge_index.to(device=self.device)
        edge_type=edge_type.to(device=self.device)
        all_intent=["chat","question","recommend"]

        batch_size=len(intent)
        intent_label=torch.zeros(batch_size,device=self.device)
        for i in range(batch_size):
            intent_label[i]=all_intent.index(intent[i])
        intent_label=intent_label.long()

        return tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2
    

     #     return rec
    def prepare_data_interactive(self,dialog_history,mention_history,edge_type,edge_index):
        #tokenized_dialog,all_length,maxlen,init_hidden = self.utter_embedder.prepare_data(dialog_history,self.device)
        tokenized_dialog,all_length,maxlen,init_hidden = self.utter_embedder.prepare_data(dialog_history,self.device)

        edge_index=edge_index.to(device=self.device)
        edge_type=edge_type.to(device=self.device)
        mention_index=[]
        mention_batch_index=[]
        for i,item in enumerate(mention_history):
            mentioned=[]
            for k in item:
                mention_index.append(k)
                mention_batch_index.append(i)
                mentioned.append(k)
            if len(mentioned)==0:
                mention_index.append(30458)
                mention_batch_index.append(i)
        mention_index=torch.Tensor(mention_index).long().to(device=self.device)
        mention_batch_index=torch.Tensor(mention_batch_index).long().to(device=self.device)
        return tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index
           
        # if self.nagetive_sample_ratio!=0:
        #     label1=torch.FloatTensor(label1).to(device=device)
        # else
    

    def prepare_data_gorecdial(self,dialog_history,mention_history,intent,node_candidate1,node_candidate2,edge_type,edge_index,label1,label2,rec_cand,all_bows):
        tokenized_dialog,all_length,maxlen,init_hidden = self.utter_embedder.prepare_data(dialog_history,self.device)
        mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,label_1,label_2=self.graph_walker.prepare_data(mention_history,intent,node_candidate1,node_candidate2,label1,label2,self.device,sample=False,dataset="gorecdial")
        edge_index=edge_index.to(device=self.device)
        rec_index,rec_batch_index,rec_golden=self.explicit_recommender.prepare_data(rec_cand,self.device)
        edge_type=edge_type.to(device=self.device)
        all_intent=["chat","question","recommend"]

        batch_size=len(intent)
        intent_label=torch.zeros(batch_size,device=self.device)
        for i in range(batch_size):
            intent_label[i]=all_intent.index(intent[i])
        intent_label=intent_label.long()

        bow_embed=all_bows.to(device=self.device)

        return tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2


    def prepare_rectest(self,rec_cand):
        sel_indices=[]
        sel_batch_indices=[]
        golden=[]
        bat=[]
        sel=[]
        grp_bat=[]
        last=[]
        for i,item in enumerate(rec_cand):
            correct=item[0]
            last.append(19308)
            grp_bat.append(i)
            item_copy=copy.deepcopy(item)
            random.shuffle(item_copy)
            #golden.append(0)
            golden.append(item_copy.index(correct))
            for k in item_copy:
                bat.append(i)
                sel.append(k)
        rec_index=torch.Tensor(sel).long().to(device=self.device)
        batch_index=torch.Tensor(bat).long().to(device=self.device)
        group_index=batch_index
        grp_bat_index=torch.Tensor(grp_bat).long().to(device=self.device)
        last_index=torch.Tensor(last).long().to(device=self.device)
        # rec_index=sel_index
        # rec_batch_index=batch_index


        batch_size=len(rec_cand)

        intent_index=[]
        for i in range(batch_size):
            intent_index.append(2)##
        intent_index=torch.Tensor(intent_index).long().to(device=self.device)
        
        return rec_index,batch_index,group_index,grp_bat_index,last_index,intent_index,golden

    def get_group_index(self,cur_index,group_name):
        group_index=[]
        for indices in cur_index:
            cur=map(int,indices[group_name])
            group_index.append(cur)
        return group_index


        
