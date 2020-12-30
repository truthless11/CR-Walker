import torch
import torch.nn as nn
from torch_scatter import scatter
from copy import deepcopy
import random



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
            result=torch.sum(alpha.mul(V)).view(1,-1)
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





class Graph_Walker(nn.Module):
    def __init__(self,device_str='cuda:1',graph_embed_size=64,utterance_embed_size=64,attention_hidden_dim=20,nagetive_sample_ratio=3):
        super(Graph_Walker,self).__init__()
        
        self.graph_embed_size=graph_embed_size
        self.utterance_embed_size=utterance_embed_size
        self.nagetive_sample_ratio=nagetive_sample_ratio

        self.context_embed_size=graph_embed_size*2+utterance_embed_size
        self.attention_hidden_dim=attention_hidden_dim
        self.parameter_list=[]

        self.device=torch.device(device_str)

        # self.W1=nn.Linear(self.context_embed_size,1)
        # self.W2=nn.Linear(self.context_embed_size,1)
        # self.W3=nn.Linear(self.context_embed_size,1)

        # self.atten1=nn.Linear(utterance_embed_size,graph_embed_size)
        # self.atten2=nn.Linear(utterance_embed_size,graph_embed_size)
        # self.atten3=nn.Linear(utterance_embed_size,graph_embed_size)

        # self.parameter_list=[[self.W1,self.atten1],[self.W2,self.atten2],[self.W3,self.atten3]]

        self.W21=nn.Linear(self.context_embed_size,1)
        self.W22=nn.Linear(self.context_embed_size,1)
        self.W23=nn.Linear(self.context_embed_size,1)

        self.W11=nn.Linear(self.context_embed_size,1)
        self.W12=nn.Linear(self.context_embed_size,1)
        self.W13=nn.Linear(self.context_embed_size,1)

        self.W21=nn.Linear(self.context_embed_size,1)
        self.W22=nn.Linear(self.context_embed_size,1)
        self.W23=nn.Linear(self.context_embed_size,1)
       
        self.parameter_list=[[self.W11,self.W12,self.W13],[self.W21,self.W22,self.W23]]

        self.abstract_embed=torch.zeros(1,self.utterance_embed_size).to(self.device)
        
        self.user_attention=Self_Attention(graph_embed_size,attention_hidden_dim)


    def get_user_portrait(self,mention_index,mention_batch_index,graph_embed):
        mention_embed=graph_embed.index_select(0,mention_index)
        #print(mention_index)
        user_portrait=self.user_attention(mention_embed,batch_index=mention_batch_index)
        return user_portrait

    def forward_single_layer(self, layer_num, utter_embed, user_portrait, graph_embed, sel_index,sel_batch_index,sel_group_index, grp_batch_index, last_index, intent_index):

        context_embed=torch.cat([utter_embed,user_portrait],dim=-1)
        #sel_index:list of length num_path, each with total_node_i*1 (total_node_i refer to all nodes within each batch for path i,i=1,2...)
        #sel_batch_index:list of length num_path, each with total_node_i*1, indicator of which batch.
        #dialog_features:list of length num_path, each with total_node_i*dialog_feature_size(to be added)
        #context_embed:batch_size*context_embed_size, use tile_context to obtain total_node_i*context_embed_size
        batch_size=context_embed.size()[0]
        #print("layer_num:",layer_num)
        num_grp=sel_group_index[-1].item()+1
        num_node=sel_index.size()[0]

        graph_embed_e=torch.cat([graph_embed,self.abstract_embed],dim=0)

        graph_features=graph_embed_e.index_select(0,sel_index)
        
        #print(context_embed.size())
        #print(sel_batch_index.size())
        tiled_context_embed=self.tile_context(context_embed,grp_batch_index)
        start_point_embed=graph_embed_e.index_select(0,last_index)
        grp_context=torch.cat([tiled_context_embed,start_point_embed],dim=-1)
        # print(i)
        weight_chat=torch.sigmoid(self.parameter_list[layer_num][0](grp_context))
        weight_question=torch.sigmoid(self.parameter_list[layer_num][1](grp_context))
        weight_recommend=torch.sigmoid(self.parameter_list[layer_num][2](grp_context))
        # weight_chat=torch.sigmoid(self.parameter_list[0][0](grp_context))
        # weight_question=torch.sigmoid(self.parameter_list[1][0](grp_context))
        # weight_recommend=torch.sigmoid(self.parameter_list[2][0](grp_context))
        #print(intent_index)
        #print(num_grp)
        weights=[]
        for i in range(num_grp):
            if intent_index[i]==0:
                weights.append(weight_chat[i,:])
            elif intent_index[i]==1:
                weights.append(weight_question[i,:])
            else:
                weights.append(weight_recommend[i,:])
        weights=torch.stack(weights)
        tiled_weights=self.tile_context(weights,sel_group_index)
        tiled_utter=self.tile_context(utter_embed,sel_batch_index)
        tiled_portrait=self.tile_context(user_portrait,sel_batch_index)

        query_vector=tiled_utter*tiled_weights+tiled_portrait*(1-tiled_weights)

        scores=torch.sum(query_vector*graph_features,dim=-1)
        # scores_chat=torch.sum(self.parameter_list[0][1](query_vector)*graph_features,dim=-1)
        # scores_question=torch.sum(self.parameter_list[1][1](query_vector)*graph_features,dim=-1)
        # scores_recommend=torch.sum(self.parameter_list[2][1](query_vector)*graph_features,dim=-1)

        # scores=torch.stack([scores_chat,scores_question,scores_recommend]).transpose(0,1).flatten()
        # linspace=3*torch.linspace(0,num_node-1).long().to(device=self.device)
        # tiled_intent=self.tile_context(intent_index,sel_group_index)
        # indexes=tiled_intent+linspace


        # # for i in range(num_node):
        # #     if intent_index[sel_group_index[i]]==0:
        # #         scores.append(scores_chat[i])
        # #     elif intent_index[sel_group_index[i]]==1:
        # #         scores.append(scores_question[i])
        # #     else:
        # #         scores.append(scores_recommend[i])
        # scores=scores.index_select(0,indexes)
        return scores
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

    def forward(self,graph_embed,utterance_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,ret_portrait=False):
        #utter_embed=#utterance_embed[:,-1,:]
        user_portrait=self.get_user_portrait(mention_index,mention_batch_index,graph_embed)
        paths=[]
        for i in range(2):
            scores=self.forward_single_layer(i,utterance_embed,user_portrait,graph_embed,sel_indices[i],sel_batch_indices[i],sel_group_indices[i],grp_batch_indices[i],last_indices[i],intent_indices[i])
                #layer_selection.append(scores)
            paths.append(scores)
        if ret_portrait:
            return paths,user_portrait
        else:
            return paths

    def prepare_data(self,mention_history,intent,node_candidate1,node_candidate2,label_1,label_2,device,sample=False,dataset="redial"):
        movie_cand=[0 for _ in range(6924)]
        all_intent=["chat","question","recommend"]
        batch_size=len(intent)
        intent_label=[0 for _ in range(batch_size)]
        for i in range(batch_size):
            intent_label[i]=all_intent.index(intent[i])
        
        sel_indices=[]
        sel_batch_indices=[]
        sel_group_indices=[]
        grp_batch_indices=[]
        last_indices=[]
        intent_indices=[]
        label1=[]
        label2=[]

        if dataset=="redial":
            bat=[]
            sel=[]
            grp=[]
            grp_bat=[]
            last=[]
            for i,item in enumerate(node_candidate1):
                last.append(30459)
                grp_bat.append(i)
                if len(item)==0:
                    my_label=[0]
                    bat.append(i)
                    grp.append(i)
                    sel.append(30458)
                elif item[0]!=-1:
                    my_label=[0 for _ in range(len(item))]
                    for lab in label_1[i]:
                        my_label[lab]=1
                    for k in item:
                        bat.append(i)
                        grp.append(i)
                        sel.append(k)
                else:
                    if sample==False:
                        my_label=deepcopy(movie_cand)
                        for lab in label_1[i]:
                            my_label[lab]=1
                        for k in range(6924):
                            bat.append(i)
                            grp.append(i)
                            sel.append(k)
                    else:
                        if self.nagetive_sample_ratio!=0:
                            positive_len=len(label_1[i])
                            negative_len=int(positive_len*self.nagetive_sample_ratio)
                            my_label=[0 for _ in range(positive_len+negative_len)]
                            for idx in range(positive_len):
                                my_label[idx]=1
                            for k in label_1[i]:
                                bat.append(i)
                                grp.append(i)
                                sel.append(k)
                            for k in range(negative_len):
                                cand=random.sample(range(6924),1)[0]
                                while cand in label_1[i]:
                                    cand=random.sample(range(6924),1)[0]
                                bat.append(i)
                                grp.append(i)
                                sel.append(cand)

                        else:
                            my_label=label_1[i]
                            for k in range(6924):
                                bat.append(i)
                                grp.append(i)
                                sel.append(k)

                label1=label1+my_label
            sel_index=torch.Tensor(sel).long().to(device=device)
            grp_index=torch.Tensor(grp).long().to(device=device)
            batch_index=torch.Tensor(bat).long().to(device=device)
            intent_index=torch.Tensor(intent_label).long().to(device=device)
            grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
            last_index=torch.Tensor(last).long().to(device=device)

            sel_indices.append(sel_index)
            sel_batch_indices.append(batch_index)
            sel_group_indices.append(grp_index)
            grp_batch_indices.append(grp_bat_index)
            last_indices.append(last_index)
            intent_indices.append(intent_index)


            bat=[]
            sel=[]
            grp=[]
            itt=[]
            grp_bat=[]
            last=[]
            grp_cnt=0
            for i,batch in enumerate(node_candidate2):
                if len(batch)==0:
                    my_label=[0]
                    label2=label2+my_label
                    bat.append(i)
                    grp.append(grp_cnt)
                    grp_bat.append(i)
                    last.append(30459)
                    sel.append(30458)
                    itt.append(intent_label[i])
                    grp_cnt+=1
                    continue
                for j,item in enumerate(batch):
                    if len(item)==0:
                        grp_bat.append(i)
                        if node_candidate1[i][0]==-1:
                            last.append(label1_[i][j])
                        else:
                            last.append(node_candidate1[i][label_1[i][j]])

                        my_label=[0]
                        label2=label2+my_label
                        bat.append(i)
                        grp.append(grp_cnt)
                        sel.append(30458)
                        itt.append(intent_label[i])
                        grp_cnt+=1
                    else:
                        grp_bat.append(i)
                        if node_candidate1[i][0]==-1:
                            last.append(label_1[i][j])
                        else:
                            last.append(node_candidate1[i][label_1[i][j]])

                        
                        my_label=[0 for _ in range(len(item))]
                        for lab in label_2[i][j]:
                            my_label[lab]=1
                        label2=label2+my_label
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
            


            sel_indices.append(sel_index)
            sel_batch_indices.append(batch_index)
            sel_group_indices.append(grp_index)
            intent_indices.append(intent_index)
            grp_batch_indices.append(grp_bat_index)
            last_indices.append(last_index)

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
        else:
            bat=[]
            sel=[]
            grp=[]
            grp_bat=[]
            last=[]
            for i,item in enumerate(node_candidate1):
                label1=label1+label_1[i]
                grp_bat.append(i)
                last.append(19308)
                for k in item:
                    bat.append(i)
                    grp.append(i)
                    sel.append(k)
            sel_index=torch.Tensor(sel).long().to(device=device)
            grp_index=torch.Tensor(grp).long().to(device=device)
            batch_index=torch.Tensor(bat).long().to(device=device)
            intent_index=torch.Tensor(intent_label).long().to(device=device)
            grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
            last_index=torch.Tensor(last).long().to(device=device)

            #print(intent_index)

            sel_indices.append(sel_index)
            sel_batch_indices.append(batch_index)
            sel_group_indices.append(grp_index)
            intent_indices.append(intent_index)
            grp_batch_indices.append(grp_bat_index)
            last_indices.append(last_index)


            bat=[]
            sel=[]
            grp=[]
            itt=[]
            grp_bat=[]
            last=[]
            grp_cnt=0
            for i,batch in enumerate(node_candidate2):
                for j,item in enumerate(batch):
                    grp_bat.append(i)
                    last.append(19308)
                    label2=label2+label_2[i][j]
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

            #print(intent_index)
            #print(grp_bat_index)

            sel_indices.append(sel_index)
            sel_batch_indices.append(batch_index)
            sel_group_indices.append(grp_index)
            intent_indices.append(intent_index)
            grp_batch_indices.append(grp_bat_index)
            last_indices.append(last_index)
            
            mention_index=[]
            mention_batch_index=[]
            #print(mention_history)
            for i,item in enumerate(mention_history):
                mentioned=[]
                for k in item:
                    mention_index.append(k)
                    mention_batch_index.append(i)
                    mentioned.append(k)
                if len(mentioned)==0:
                    mention_index.append(19307)
                    mention_batch_index.append(i)
            # for i,batch in enumerate(mention_history):
            #     mentioned=[]
            #     for item in batch:
            #         for k in item:
            #             if k not in mentioned:
            #                 mention_index.append(k)
            #                 mention_batch_index.append(i)
            #                 mentioned.append(k)
            #     if len(mentioned)==0:
            #         mention_index.append(19307)
            #         mention_batch_index.append(i)
           
        # if self.nagetive_sample_ratio!=0:
        #     label1=torch.FloatTensor(label1).to(device=device)
        # else:
        mention_index=torch.Tensor(mention_index).long().to(device=device)
        mention_batch_index=torch.Tensor(mention_batch_index).long().to(device=device)
        label1=torch.FloatTensor(label1).to(device=device)
        label2=torch.FloatTensor(label2).to(device=device)
        
        return mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,label1,label2
    

            
        #context_embed(list)



