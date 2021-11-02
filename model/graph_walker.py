import torch
import torch.nn as nn
from torch_scatter import scatter
from copy import deepcopy
from conf import args,add_generic_args
import random
import os.path as osp
import sys
sys.path.append("..")
from data.redial import ReDial
from data.gorecdial import GoRecDial
from torch_geometric.data import DataLoader



class Attention(nn.Module):
    #beta=tanh(QW+KU)*V
    def __init__(self,Q_size,K_size,atten_hidden):
        super(Attention,self).__init__()
        self.Q_size=Q_size
        self.K_size=K_size
        self.Attention_W=nn.Linear(Q_size,atten_hidden,bias=False)
        self.Attention_U=nn.Linear(K_size,atten_hidden,bias=False)
        self.Attention_V=nn.Linear(atten_hidden,1,bias=False)
   
    def forward(self,Q,K,V,batch_index=None):
        #Q,K,V:(num_nodes,*)
        QW=self.Attention_W(Q)
        KU=self.Attention_U(K)
        beta=self.Attention_V(torch.tanh(QW+KU)).squeeze()#[46]
        if batch_index==None:
            alpha=torch.softmax(beta)
            result=torch.sum(alpha.mul(V)).view(1,-1)
        else:
            exp_beta=torch.exp(beta)
            grouped_sum=scatter(exp_beta,batch_index,dim=0,reduce='sum')
            #print(batch_index.size())
            tiled_sum=self.tile_sum(grouped_sum,batch_index).squeeze()
            alpha=exp_beta.div(tiled_sum).view(-1,1)
            result=scatter(alpha.mul(V),batch_index,dim=0,reduce='sum')
            #(batch_size,V_size)
        return result
    
    def tile_sum(self,batch_sum,batch_index):
        ones=torch.ones_like(batch_index)#.cuda(device=torch.device('cuda:1'))
        graph_size=scatter(ones,batch_index,dim=0,reduce='sum')
        #turn_utter(batch_size,rnn_hidden)
        #graph_size(batch_size)
        batch_size=graph_size.size()[0]
        tile_sum=[]
        for i in range(batch_size):
            repeated=batch_sum[i].repeat(graph_size[i],1)
            tile_sum.append(repeated)
        tile_sum=torch.cat(tile_sum,dim=0)
        return tile_sum



class Dot_Attention(nn.Module):
    #beta=tanh(QW+KU)*V
    def __init__(self,Q_size,K_size,atten_hidden):
        super(Attention,self).__init__()
        self.Q_size=Q_size
        self.K_size=K_size
        self.Attention_W=nn.Linear(Q_size,K_size,bias=False)
   
    def forward(self,Q,K,V,batch_index=None):
        beta=(self.Attention_W(Q)*K).squeeze()#[46]
        if batch_index==None:
            alpha=torch.softmax(beta)
            result=torch.sum(alpha.mul(V)).view(1,-1)
        else:
            exp_beta=torch.exp(beta)
            grouped_sum=scatter(exp_beta,batch_index,dim=0,reduce='sum')
            #print(batch_index.size())
            tiled_sum=self.tile_sum(grouped_sum,batch_index).squeeze()
            alpha=exp_beta.div(tiled_sum).view(-1,1)
            result=scatter(alpha.mul(V),batch_index,dim=0,reduce='sum')
            #(batch_size,V_size)
        return result,alpha
    
    def tile_sum(self,batch_sum,batch_index):
        ones=torch.ones_like(batch_index)#.cuda(device=torch.device('cuda:1'))
        graph_size=scatter(ones,batch_index,dim=0,reduce='sum')
        #turn_utter(batch_size,rnn_hidden)
        #graph_size(batch_size)
        batch_size=graph_size.size()[0]
        tile_sum=[]
        for i in range(batch_size):
            repeated=batch_sum[i].repeat(graph_size[i],1)
            tile_sum.append(repeated)
        tile_sum=torch.cat(tile_sum,dim=0)
        return tile_sum






class Self_Attention(nn.Module):
    def __init__(self,embed_size,atten_hidden):
        super(Self_Attention,self).__init__()
        self.embed_size=embed_size
        self.atten_hidden=atten_hidden
        self.Attention_W=nn.Linear(embed_size,atten_hidden,bias=False)
        self.Attention_V=nn.Linear(atten_hidden,1,bias=False)

    def forward(self,embed,batch_index=None):
        #input:(num_nodes,embed_size)
        beta=self.Attention_V(torch.tanh(self.Attention_W(embed))).squeeze(-1)
        if batch_index==None:
            alpha=torch.softmax(beta)
            result=torch.sum(alpha.mul(embed)).view(1,-1)
        else:
            exp_beta=torch.exp(beta)
            #print(exp_beta)
            #print(batch_index)
            grouped_sum=scatter(exp_beta,batch_index,dim=0,reduce='sum')
            #print(batch_index.size())
            #print("grouped_sum:",grouped_sum.size())
            tiled_sum=self.tile_sum(grouped_sum,batch_index).squeeze()
            #print("tiled_sum:",tiled_sum.size())
            alpha=exp_beta.div(tiled_sum).view(-1,1)
            result=scatter(alpha.mul(embed),batch_index,dim=0,reduce='sum')
            #(batch_size,embed_size)
        return result

    def tile_sum(self,batch_sum,batch_index):
        ones=torch.ones_like(batch_index)
        graph_size=scatter(ones,batch_index,dim=0,reduce='sum')
        batch_size=graph_size.size()[0]
        tile_sum=[]
        for i in range(batch_size):
            repeated=batch_sum[i].repeat(graph_size[i],1)
            tile_sum.append(repeated)
        tile_sum=torch.cat(tile_sum,dim=0)
        return tile_sum





class Graph_Walker(nn.Module):
    def __init__(self,device_str='cuda:1',graph_embed_size=64,utterance_embed_size=64,attention_hidden_dim=20,nagetive_sample_ratio=3,G_pseudo_sample_num=3,P_pseudo_sample_num=5,word_net=False):
        super(Graph_Walker,self).__init__()
        
        self.graph_embed_size=graph_embed_size
        self.word_embed_size=graph_embed_size
        self.utterance_embed_size=utterance_embed_size
        self.nagetive_sample_ratio=nagetive_sample_ratio
        self.G_pseudo_sample_num=G_pseudo_sample_num
        self.P_pseudo_sample_num=P_pseudo_sample_num

        self.context_embed_size=graph_embed_size*2+2*utterance_embed_size
        self.attention_hidden_dim=attention_hidden_dim
        self.parameter_list=[]

        self.bow_size=512

        self.word_net=word_net

        self.device=torch.device(device_str)

        self.bow_embed=None

        self.Wu=nn.Linear(2*graph_embed_size,1)
        self.W1=nn.Linear(self.context_embed_size,1)
        self.W2=nn.Linear(self.context_embed_size,1)
       
        self.parameter_list=[self.W1,self.W2]

        self.abstract_embed=torch.zeros(1,self.utterance_embed_size).to(self.device)
        self.intent_embed=nn.Embedding(3,utterance_embed_size)
        
        self.user_attention=Self_Attention(graph_embed_size,attention_hidden_dim)

        self.word_attention=Self_Attention(graph_embed_size,attention_hidden_dim)

    
    def add_bow(self,bow_embed):
        self.bow_embed=bow_embed.to(device=self.device)


    def get_user_portrait(self,mention_index,mention_batch_index,graph_embed,word_index=None,word_batch_index=None,word_embed=None):
        mention_embed=graph_embed.index_select(0,mention_index)
        #print(mention_index)
        user_portrait=self.user_attention(mention_embed,batch_index=mention_batch_index)
        if not self.word_net:
            return user_portrait
        else:
            mention_word_embed=word_embed.index_select(0,word_index)
            word_portrait=self.word_attention(mention_word_embed,batch_index=word_batch_index)
            #print(word_batch_index)
            weight=torch.sigmoid(self.Wu(torch.cat([user_portrait,word_portrait],dim=-1)))
            final_portrait=user_portrait*weight+word_portrait*(1-weight)
            return final_portrait


    def forward_single_layer(self, layer_num, utter_embed, user_portrait, graph_embed, sel_index,sel_batch_index,sel_group_index, grp_batch_index, last_index, intent_index,score_mask,last_weight=None,ret_partial_score=False):

        context_embed=torch.cat([utter_embed,user_portrait],dim=-1)
        batch_size=context_embed.size()[0]
        num_node=sel_index.size()[0]
        
        graph_embed_e=torch.cat([graph_embed,self.abstract_embed],dim=0)

        graph_features=graph_embed_e.index_select(0,sel_index)
        tiled_context_embed=self.tile_context(context_embed,grp_batch_index)
        start_point_embed=graph_embed_e.index_select(0,last_index)
        itt_embed=self.intent_embed(intent_index)
        grp_context=torch.cat([tiled_context_embed,start_point_embed,itt_embed],dim=-1)
        weights=torch.sigmoid(self.parameter_list[layer_num](grp_context))

        tiled_weights=self.tile_context(weights,sel_group_index)
        tiled_utter=self.tile_context(utter_embed,sel_batch_index)
        tiled_portrait=self.tile_context(user_portrait,sel_batch_index)

        query_vector=tiled_utter*tiled_weights+tiled_portrait*(1-tiled_weights)

        scores=torch.sum(query_vector*graph_features,dim=-1)

        if last_weight!=None:
            tiled_last_weight=self.tile_context(last_weight,sel_batch_index)
            last_query_vector=tiled_utter*tiled_last_weight+tiled_portrait*(1-tiled_last_weight)
            last_scores=torch.sum(last_query_vector*graph_features,dim=-1)
            partial_score=score_mask*scores
            final_score=last_scores+partial_score
            if ret_partial_score:
                return final_score,None,partial_score
            else:
                return final_score,None
        else:
            if ret_partial_score:
                return scores,weights,None
            else:
                return scores,weights
                
    def tile_context(self,context,batch_index):
        ones=torch.ones_like(batch_index).to(device=self.device)
        graph_size=scatter(ones,batch_index,dim=0,reduce='sum')
        batch_size=graph_size.size()[0]
        tile_ctx=[]
        for i in range(batch_size):
            repeated=context[i].repeat(graph_size[i],1)
            tile_ctx.append(repeated)
        tile_ctx=torch.cat(tile_ctx,dim=0)
        return tile_ctx

    def forward(self,graph_embed,utterance_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,score_masks,ret_portrait=False,word_embed=None,word_batch_index=None,word_index=None):
        #utter_embed=#utterance_embed[:,-1,:]
        user_portrait=self.get_user_portrait(mention_index,mention_batch_index,graph_embed,word_index,word_batch_index,word_embed)
        paths=[]
        last_weight=None
        for i in range(2):
            scores,last_weight=self.forward_single_layer(i,utterance_embed,user_portrait,graph_embed,sel_indices[i],sel_batch_indices[i],sel_group_indices[i],grp_batch_indices[i],last_indices[i],intent_indices[i],score_masks[i],last_weight)
                #layer_selection.append(scores)
            paths.append(scores)
        if ret_portrait:
            return paths,user_portrait
        else:
            return paths

    def prepare_data(self,mention_history,intent,node_candidate1,node_candidate2,label_1,label_2,attribute_dict,device,gold_pos=None,sample=False,dataset='gorecdial'):
        movie_cand=[0 for _ in range(6924)]
        all_intent=["chat","question","recommend"]
        batch_size=len(intent)
        intent_label=[0 for _ in range(batch_size)]
        for i in range(batch_size):
            intent_label[i]=all_intent.index(intent[i])
        
        null_idx=30458 if dataset=="redial" else 19307

        
        sel_indices=[]
        sel_batch_indices=[]
        sel_group_indices=[]
        grp_batch_indices=[]
        last_indices=[]
        intent_indices=[]
        label1=[]
        label2=[]
        score_masks=[]


        bat=[]
        sel=[]
        grp=[]
        mask=[]
        grp_bat=[]
        last=[]

        for i,item in enumerate(node_candidate1):
            last.append(null_idx)
            grp_bat.append(i)
            if len(item)==0:
                my_label=[0]
                my_mask=[1]
                bat.append(i)
                grp.append(i)
                sel.append(null_idx)
            elif intent_label[i]==2:
                my_label=[0 for _ in range(len(item))]
                my_mask=[1 for _ in range(len(item))]
                all_pseudo=[]
                is_gold=0
                for p,lab in enumerate(label_1[i]):
                    if gold_pos!=None:
                        if gold_pos[i][p]==1:
                            my_label[lab]=1
                            #all_pseudo.append(lab)
                            is_gold=1
                        else:
                            all_pseudo.append(lab)
                    else:
                        all_pseudo.append(lab)

                if is_gold:
                    sampled=random.sample(all_pseudo,min(len(all_pseudo),self.G_pseudo_sample_num))
                else:
                    sampled=random.sample(all_pseudo,min(len(all_pseudo),self.P_pseudo_sample_num))
                for lab in sampled:
                    my_label[lab]=1
                for k in item:
                    bat.append(i)
                    grp.append(i)
                    sel.append(k)
            else:
                my_label=[0 for _ in range(len(item))]
                my_mask=[1 for _ in range(len(item))]
                for lab in label_1[i]:
                    my_label[lab]=1
                for k in item:
                    bat.append(i)
                    grp.append(i)
                    sel.append(k)

            label1=label1+my_label
            mask=mask+my_mask
        sel_index=torch.Tensor(sel).long().to(device=device)
        grp_index=torch.Tensor(grp).long().to(device=device)
        batch_index=torch.Tensor(bat).long().to(device=device)
        intent_index=torch.Tensor(intent_label).long().to(device=device)
        grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
        last_index=torch.Tensor(last).long().to(device=device)
        score_mask=torch.Tensor(mask).float().to(device=device)

        sel_indices.append(sel_index)
        sel_batch_indices.append(batch_index)
        sel_group_indices.append(grp_index)
        grp_batch_indices.append(grp_bat_index)
        last_indices.append(last_index)
        intent_indices.append(intent_index)
        score_masks.append(score_mask)

        bat=[]
        sel=[]
        grp=[]
        itt=[]
        grp_bat=[]
        last=[]
        mask=[]
        grp_cnt=0


        for i,batch in enumerate(node_candidate2):
            if len(batch)==0:
                my_label=[0]
                my_mask=[1]
                label2=label2+my_label
                mask=mask+my_mask
                bat.append(i)
                grp.append(grp_cnt)
                grp_bat.append(i)
                last.append(null_idx)
                sel.append(null_idx)
                itt.append(intent_label[i])
                grp_cnt+=1
                continue
            for j,item in enumerate(batch):
                if j>=len(label_1[i]):
                    break
                if len(item)==0:
                    grp_bat.append(i)
                    last.append(node_candidate1[i][label_1[i][j]])
                    my_label=[0]
                    my_mask=[1]
                    label2=label2+my_label
                    mask=mask+my_mask
                    bat.append(i)
                    grp.append(grp_cnt)
                    sel.append(null_idx)
                    itt.append(intent_label[i])
                    grp_cnt+=1
                elif item[0]==-1:
                    grp_bat.append(i)
                    last.append(node_candidate1[i][label_1[i][j]])
                    positive_len=len(label_2[i][j])
                    negative_len=int(positive_len*self.nagetive_sample_ratio)
                    my_label=[0 for _ in range(positive_len+negative_len)]
                    my_mask=[0 for _ in range(positive_len+negative_len)]
                    for idx in range(positive_len):
                        my_label[idx]=1
                    for idx,k in enumerate(label_2[i][j]):
                        bat.append(i)
                        grp.append(grp_cnt)
                        sel.append(k)
                        if k in attribute_dict[node_candidate1[i][label_1[i][j]]]:
                            my_mask[idx]=1
                    for k in range(negative_len):
                        cand=random.sample(range(6924),1)[0]
                        while cand in label_2[i][j]:
                            cand=random.sample(range(6924),1)[0]
                        bat.append(i)
                        grp.append(grp_cnt)
                        sel.append(cand)
                        if cand in attribute_dict[node_candidate1[i][label_1[i][j]]]:
                            my_mask[positive_len+k]=1
                    label2=label2+my_label
                    mask=mask+my_mask
                    itt.append(intent_label[i])
                    grp_cnt+=1
                else:
                    grp_bat.append(i)
                    last.append(node_candidate1[i][label_1[i][j]])
                    my_mask=[1 for _ in range(len(item))]
                    my_label=[0 for _ in range(len(item))]
                    for lab in label_2[i][j]:
                        my_label[lab]=1
                    label2=label2+my_label
                    mask=mask+my_mask
                    for k in item:
                        bat.append(i)
                        grp.append(grp_cnt)
                        sel.append(k)
                    itt.append(intent_label[i])
                    grp_cnt+=1

        sel_index=torch.Tensor(sel).long().to(device=device)
        grp_index=torch.Tensor(grp).long().to(device=device)
        batch_index=torch.Tensor(bat).long().to(device=device)
        intent_index=torch.Tensor(itt).long().to(device=device)
        grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
        last_index=torch.Tensor(last).long().to(device=device)
        score_mask=torch.Tensor(mask).float().to(device=device)

        sel_indices.append(sel_index)
        sel_batch_indices.append(batch_index)
        sel_group_indices.append(grp_index)
        grp_batch_indices.append(grp_bat_index)
        last_indices.append(last_index)
        intent_indices.append(intent_index)
        score_masks.append(score_mask)


        mention_index=[]
        mention_batch_index=[]
        
        for i,item in enumerate(mention_history):
            mentioned=[]
            for k in item:
                mention_index.append(k)
                mention_batch_index.append(i)
                mentioned.append(k)
            if len(mentioned)==0:
                mention_index.append(null_idx)
                mention_batch_index.append(i)
        
        mention_index=torch.Tensor(mention_index).long().to(device=device)
        mention_batch_index=torch.Tensor(mention_batch_index).long().to(device=device)
        label1=torch.FloatTensor(label1).to(device=device)
        label2=torch.FloatTensor(label2).to(device=device)

        

        return mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,label1,label2,score_masks
        
     