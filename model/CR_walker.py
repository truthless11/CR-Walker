
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
    def __init__(self,device_str='cuda:1',rnn_type="RNN_TANH",use_bert=True,utter_embed_size=64,dropout=0.5,num_turns=10,num_relations=12,num_bases=15,graph_embed_size=64,atten_hidden=20,negative_sample_ratio=5,dataset="redial",word_net=False):
        super(ProRec,self).__init__()
        self.dataset=dataset
        if dataset=="redial":
            self.null_idx=30458
            self.num_nodes=30471
        else:
            self.null_idx=19307
            self.num_nodes=19308
        
        self.num_words=29308
        self.use_bert=use_bert
        self.utter_embed_size=utter_embed_size
        self.dropout=dropout
        self._lambda1=0.1
        self._lambda2=0.01
        #self.maxlen=maxlen
        self.word_net=word_net
        
        self.num_turns=num_turns
        self.num_relations=num_relations
        self.num_bases=num_bases
        self.graph_embed_size=graph_embed_size
        self.atten_hidden=atten_hidden
        

        self.device=torch.device(device_str)
        self.negative_sample_ratio=negative_sample_ratio

        self.utter_embedder=Utterance_Embedder(rnn_type,use_bert,utter_embed_size,dropout,num_turns,word_net=word_net)
        self.graph_embedder=Graph_Embedder(num_nodes=self.num_nodes,embed_size=graph_embed_size,device_str=device_str,word_net=word_net)
        self.intent_selector=IntentSelector(utter_embed_size,atten_hidden)
        self.graph_walker=Graph_Walker(attention_hidden_dim=atten_hidden,graph_embed_size=graph_embed_size,utterance_embed_size=utter_embed_size,device_str=device_str,nagetive_sample_ratio=negative_sample_ratio,word_net=word_net)
        if self.dataset=="gorecdial":
            self.explicit_recommender=Explicit_Recommender(utterance_embed_size=utter_embed_size,graph_embed_size=graph_embed_size)#utterance_embed_size=utter_embed_size,graph_embed_size=graph_embed_size)


        self.walk_loss_1=torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.walk_loss_2=nn.BCEWithLogitsLoss(reduction="sum")
        self.intent_loss=torch.nn.CrossEntropyLoss(reduction='sum')
        self.rec_loss=torch.nn.CrossEntropyLoss(reduction='sum')

        self.alignment_loss=nn.BCEWithLogitsLoss(reduction="sum")
        self.alignment_loss_word=nn.BCEWithLogitsLoss(reduction="sum")
        #self.alignment_loss=nn.MSELoss()
        self.Wa=nn.Linear(utter_embed_size,graph_embed_size,bias=False)
        self.Ww=nn.Linear(utter_embed_size,graph_embed_size,bias=False)
        #self.Wo=nn.Linear(1,self.num_nodes)



    def forward_pretrain(self,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label,alignment_index_word=None,alignment_batch_index_word=None,alignment_label_word=None):
        
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #last_utter=utter_embed[:,-1,:]
        #print(last_utter)
        intent=self.intent_selector.forward(utter_embed)
        #print(alignment_index)
        graph_embed,word_embed=self.graph_embedder.forward(edge_type,edge_index)
        graph_features=graph_embed.index_select(0,alignment_index)
        
        tiled_utter=self.graph_walker.tile_context(utter_embed,alignment_batch_index)
        logits=torch.sum(self.Wa(tiled_utter)*graph_features,dim=-1)

        loss_a=self.alignment_loss(logits,alignment_label)
        intent_loss=self.intent_loss(intent,intent_label)

        if self.word_net:
            tiled_utter_word=self.graph_walker.tile_context(utter_embed,alignment_batch_index_word)
            word_features=word_embed.index_select(0,alignment_index_word)
            logits_w=torch.sum(self.Ww(tiled_utter_word)*word_features,dim=-1)
            loss_b=self.alignment_loss_word(logits_w,alignment_label_word)
            return loss_a+loss_b+intent_loss
        else:
            return loss_a+intent_loss

    
    def prepare_reg(self,mention_history,dialog_history,intent=None,rec_cand=None):
        alignment_index=[]
        alignment_batch_index=[]
        alignment_label=[]
        for i,item in enumerate(mention_history):
            mentioned=[]
            for k in item:
                if rec_cand!=None and k in rec_cand[i]:
                    continue
                alignment_index.append(k)
                alignment_batch_index.append(i)
                alignment_label.append(1)
                mentioned.append(k)
            len_positive=len(mentioned)

            if rec_cand!=None:
                alignment_index.append(rec_cand[i][0])
                alignment_batch_index.append(i)
                alignment_label.append(1)


            if len(mentioned)==0:
                len_positive+=1
                mentioned.append(self.null_idx)
                alignment_index.append(self.null_idx)
                alignment_batch_index.append(i)
                alignment_label.append(1)
            else:
                alignment_index.append(self.null_idx)
                alignment_batch_index.append(i)
                alignment_label.append(0)

            
            sampled=[]

            if rec_cand!=None:
                for neg_rec in range(1,5):
                    sampled.append(rec_cand[i][neg_rec])
                    alignment_index.append(rec_cand[i][neg_rec])
                    alignment_batch_index.append(i)
                    alignment_label.append(0)
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

        

        #if intent!=None:

        alignment_index_word=None
        alignment_batch_index_word=None
        alignment_label_word=None

       

        if self.word_net:
            tokenized_dialog,all_length,maxlen,init_hidden,word_index,word_batch_index,raw_history = self.utter_embedder.prepare_data(dialog_history,self.device,raw_history=True)
            alignment_index_word=[]
            alignment_batch_index_word=[]
            alignment_label_word=[]
            for i,item in enumerate(raw_history):
                mentioned=[]
                for k in item:
                    alignment_index_word.append(k)
                    alignment_batch_index_word.append(i)
                    alignment_label_word.append(1)
                    mentioned.append(k)
                len_positive=len(mentioned)
                if len(mentioned)==0:
                    len_positive+=1
                    mentioned.append(0)
                    alignment_index_word.append(0)
                    alignment_batch_index_word.append(i)
                    alignment_label_word.append(1)
                
                for ite in range(len_positive):
                    #cand=0
                    cand=random.sample(range(self.num_words),1)[0]
                    while cand in mentioned or cand in sampled:
                        cand=random.sample(range(self.num_words),1)[0]
                    sampled.append(cand)
                    alignment_index_word.append(cand)
                    alignment_batch_index_word.append(i)
                    alignment_label_word.append(0)
            alignment_index_word=torch.Tensor(alignment_index_word).long().to(device=self.device)
            alignment_batch_index_word=torch.Tensor(alignment_batch_index_word).long().to(device=self.device)
            alignment_label_word=torch.FloatTensor(alignment_label_word).to(device=self.device)
        
        return alignment_index,alignment_batch_index,alignment_label,alignment_index_word,alignment_batch_index_word,alignment_label_word

    
    def prepare_pretrain(self,mention_history,dialog_history,intent,edge_type,edge_index,rec_cand=None):
        alignment_index=[]
        alignment_batch_index=[]
        alignment_label=[]
        for i,item in enumerate(mention_history):
            mentioned=[]
            for k in item:
                if rec_cand!=None and k in rec_cand[i]:
                    continue
                alignment_index.append(k)
                alignment_batch_index.append(i)
                alignment_label.append(1)
                mentioned.append(k)
            len_positive=len(mentioned)

            if rec_cand!=None:
                alignment_index.append(rec_cand[i][0])
                alignment_batch_index.append(i)
                alignment_label.append(1)


            if len(mentioned)==0:
                len_positive+=1
                mentioned.append(self.null_idx)
                alignment_index.append(self.null_idx)
                alignment_batch_index.append(i)
                alignment_label.append(1)
            else:
                alignment_index.append(self.null_idx)
                alignment_batch_index.append(i)
                alignment_label.append(0)

            
            sampled=[]

            if rec_cand!=None:
                for neg_rec in range(1,5):
                    sampled.append(rec_cand[i][neg_rec])
                    alignment_index.append(rec_cand[i][neg_rec])
                    alignment_batch_index.append(i)
                    alignment_label.append(0)
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

        

        #if intent!=None:
        tokenized_dialog,all_length,maxlen,init_hidden,word_index,word_batch_index,raw_history = self.utter_embedder.prepare_data(dialog_history,self.device,raw_history=True)
        edge_index=edge_index.to(device=self.device)
        edge_type=edge_type.to(device=self.device)
        all_intent=["chat","question","recommend"]
        batch_size=len(intent)
        intent_label=torch.zeros(batch_size,device=self.device)
        for i in range(batch_size):
            intent_label[i]=all_intent.index(intent[i])
        intent_label=intent_label.long()


        alignment_index_word=None
        alignment_batch_index_word=None
        alignment_label_word=None

       

        if self.word_net:
            alignment_index_word=[]
            alignment_batch_index_word=[]
            alignment_label_word=[]
            for i,item in enumerate(raw_history):
                mentioned=[]
                for k in item:
                    alignment_index_word.append(k)
                    alignment_batch_index_word.append(i)
                    alignment_label_word.append(1)
                    mentioned.append(k)
                len_positive=len(mentioned)
                if len(mentioned)==0:
                    len_positive+=1
                    mentioned.append(0)
                    alignment_index_word.append(0)
                    alignment_batch_index_word.append(i)
                    alignment_label_word.append(1)
                
                for ite in range(len_positive):
                    #cand=0
                    cand=random.sample(range(self.num_words),1)[0]
                    while cand in mentioned or cand in sampled:
                        cand=random.sample(range(self.num_words),1)[0]
                    sampled.append(cand)
                    alignment_index_word.append(cand)
                    alignment_batch_index_word.append(i)
                    alignment_label_word.append(0)
            alignment_index_word=torch.Tensor(alignment_index_word).long().to(device=self.device)
            alignment_batch_index_word=torch.Tensor(alignment_batch_index_word).long().to(device=self.device)
            alignment_label_word=torch.FloatTensor(alignment_label_word).to(device=self.device)

            
            
            
        
        return tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label,alignment_index_word,alignment_batch_index_word,alignment_label_word

    
    def forward(self,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,score_masks,alignment_index,alignment_batch_index,alignment_label,word_index=None,word_batch_index=None,alignment_index_word=None,alignment_batch_index_word=None,alignment_label_word=None):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed,word_embed=self.graph_embedder.forward(edge_type,edge_index)
        #print("graph_embedding:",graph_embed.size())
        intent=self.intent_selector.forward(utter_embed)#,graph_embeddings,mention_history,self.get_group_index(cur_type,"Candidate"),self.get_group_index(cur_type,"Attr"),graph_size)
        #print("intent:",intent.size())
        paths=self.graph_walker.forward(graph_embed,utter_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,score_masks,word_embed=word_embed,word_batch_index=word_batch_index,word_index=word_index)
        walk_loss_1=self.walk_loss_1(paths[0],label_1)
        walk_loss_2=self.walk_loss_2(paths[1],label_2)
        intent_loss=self.intent_loss(intent,intent_label)

        graph_features=graph_embed.index_select(0,alignment_index)
        tiled_utter=self.graph_walker.tile_context(utter_embed,alignment_batch_index)
        logits=torch.sum(self.Wa(tiled_utter)*graph_features,dim=-1)
        reg_loss=self.alignment_loss(logits,alignment_label)
        if self.word_net:
            word_features=word_embed.index_select(0,alignment_index_word)
            tiled_utter_word=self.graph_walker.tile_context(utter_embed,alignment_batch_index_word)
            logits_w=torch.sum(self.Ww(tiled_utter_word)*word_features,dim=-1)
            loss_b=self.alignment_loss_word(logits_w,alignment_label_word)
            reg_loss=reg_loss+loss_b


        #print(logits)
        #logits=logits+self.Wo.bias.index_select(0,alignment_index)
        

        tot_loss=walk_loss_1+walk_loss_2+intent_loss+0.025*reg_loss#
        return intent,paths,tot_loss

    
    def forward_gorecdial(self,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2,score_masks,alignment_index,alignment_batch_index,alignment_label,word_index=None,word_batch_index=None,alignment_index_word=None,alignment_batch_index_word=None,alignment_label_word=None):

        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed,word_embed=self.graph_embedder.forward(edge_type,edge_index)
        intent=self.intent_selector.forward(utter_embed)
        #print("graph_embedding:",graph_embed.size())
        #,graph_embeddings,mention_history,self.get_group_index(cur_type,"Candidate"),self.get_group_index(cur_type,"Attr"),graph_size)
        #print("intent:",intent.size())
        #paths=self.graph_walker.forward(graph_embed,utter_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,intent_indices,node_feature)
        paths,user_portrait=self.graph_walker.forward(graph_embed,utter_embed,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,score_masks,ret_portrait=True,word_embed=word_embed,word_batch_index=word_batch_index,word_index=word_index)

        

        rec=self.explicit_recommender.forward(utter_embed,graph_embed,user_portrait,bow_embed,rec_index,rec_batch_index)

        #for i,item in enumerate(paths):
            #print("hop "+str(i)+":",item.size())
        #loss1=self.walk_loss.forward(graph_embed,paths,bad_indices,batch_indices,golden_indices)
        walk_loss_1=self.walk_loss_1(paths[0],label_1)
        walk_loss_2=self.walk_loss_2(paths[1],label_2)
        intent_loss=self.intent_loss(intent,intent_label)
        rec_loss=self.rec_loss(rec,rec_golden)

        graph_features=graph_embed.index_select(0,alignment_index)
        tiled_utter=self.graph_walker.tile_context(utter_embed,alignment_batch_index)
        logits=torch.sum(self.Wa(tiled_utter)*graph_features,dim=-1)
        reg_loss=self.alignment_loss(logits,alignment_label)
        if self.word_net:
            word_features=word_embed.index_select(0,alignment_index_word)
            tiled_utter_word=self.graph_walker.tile_context(utter_embed,alignment_batch_index_word)
            logits_w=torch.sum(self.Ww(tiled_utter_word)*word_features,dim=-1)
            loss_b=self.alignment_loss_word(logits_w,alignment_label_word)
            reg_loss=reg_loss+loss_b
        #print(rec_golden.unsqueeze(-1))
        #one_hot_golden = torch.zeros(10, 5).cuda(device=self.device).scatter_(1, rec_golden.unsqueeze(-1), 1)
        #print(one_hot_golden)
        

        tot_loss=walk_loss_1+walk_loss_2+intent_loss+rec_loss+0.025*reg_loss
        return intent,paths,tot_loss



    def get_intent(self,tokenized_dialog,all_length,maxlen,init_hidden):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        intent=self.intent_selector.forward(utter_embed)
        return intent
        

    def inference_gorecdial(self,intent,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,score_mask,word_index,word_batch_index,bow_embed=None,sel_index_ex=None,sel_batch_index_ex=None,last_weights=None,layer=0):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed,word_embed=self.graph_embedder.forward(edge_type,edge_index)

        user_portrait=self.graph_walker.get_user_portrait(mention_index,mention_batch_index,graph_embed,word_index,word_batch_index,word_embed)

        step,weight,partial_score=self.graph_walker.forward_single_layer(layer,utter_embed,user_portrait,graph_embed,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,intent,score_mask,last_weights,True)
        if bow_embed!=None:
            rec=self.explicit_recommender.forward(utter_embed,graph_embed,user_portrait,bow_embed,sel_index_ex,sel_batch_index_ex)
            return step,weight,partial_score,rec
        else:
            return step,weight,partial_score
    
    def inference_redial(self,intent,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,score_mask,word_index,word_batch_index,last_weights=None,layer=0):
        utter_embed=self.utter_embedder.forward(tokenized_dialog,all_length,maxlen,init_hidden)
        #print("utterance_embedding:",utter_embed.size())
        graph_embed,word_embed=self.graph_embedder.forward(edge_type,edge_index)

        user_portrait=self.graph_walker.get_user_portrait(mention_index,mention_batch_index,graph_embed,word_index,word_batch_index,word_embed)

        step,weight,partial_score=self.graph_walker.forward_single_layer(layer,utter_embed,user_portrait,graph_embed,sel_index,sel_batch_index,sel_group_index,grp_batch_index,last_index,intent,score_mask,last_weights,ret_partial_score=True)
        return step,weight,partial_score

    def prepare_data_redial(self,dialog_history,mention_history,intent,node_candidate1,node_candidate2,edge_type,edge_index,label1,label2,gold_pos,attribute_dict,sample=False):
        tokenized_dialog,all_length,maxlen,init_hidden,word_index,word_batch_index = self.utter_embedder.prepare_data(dialog_history,self.device)
        mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,label_1,label_2,score_masks=self.graph_walker.prepare_data(mention_history,intent,node_candidate1,node_candidate2,label1,label2,attribute_dict,self.device,gold_pos,sample=sample,dataset="redial")
        edge_index=edge_index.to(device=self.device)
        edge_type=edge_type.to(device=self.device)
        all_intent=["chat","question","recommend"]

        
        batch_size=len(intent)
        intent_label=torch.zeros(batch_size,device=self.device)
        for i in range(batch_size):
            intent_label[i]=all_intent.index(intent[i])
        intent_label=intent_label.long()

        return tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,score_masks,word_index,word_batch_index
    

     #     return rec
    def prepare_data_interactive(self,dialog_history,mention_history,edge_type,edge_index):
        #tokenized_dialog,all_length,maxlen,init_hidden = self.utter_embedder.prepare_data(dialog_history,self.device)
        tokenized_dialog,all_length,maxlen,init_hidden,_,_ = self.utter_embedder.prepare_data(dialog_history,self.device)

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
    

    def prepare_data_gorecdial(self,dialog_history,mention_history,intent,node_candidate1,node_candidate2,edge_type,edge_index,label1,label2,attribute_dict,rec_cand,all_bows):
        tokenized_dialog,all_length,maxlen,init_hidden,word_index,word_batch_index= self.utter_embedder.prepare_data(dialog_history,self.device)
        mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,label_1,label_2,score_masks=self.graph_walker.prepare_data(mention_history,intent,node_candidate1,node_candidate2,label1,label2,attribute_dict,self.device,sample=False,dataset="gorecdial")
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

        return tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2,score_masks,word_index,word_batch_index


    def prepare_rectest(self,rec_cand):
        golden=[]
        bat=[]
        sel=[]
        shuffled_rec_cand=[]
        for i,item in enumerate(rec_cand):
            correct=item[0]
            item_copy=copy.deepcopy(item)
            random.shuffle(item_copy)
            golden.append(item_copy.index(correct))
            shuffled_rec_cand.append(item_copy)
            for k in item_copy:
                bat.append(i)
                sel.append(k)
        rec_index=torch.Tensor(sel).long().to(device=self.device)
        batch_index=torch.Tensor(bat).long().to(device=self.device)
        batch_size=len(rec_cand)

        intent_index=[]
        for i in range(batch_size):
            intent_index.append(2)
        intent_index=torch.Tensor(intent_index).long().to(device=self.device)
        
        return intent_index,rec_index,batch_index,shuffled_rec_cand,golden

    def get_group_index(self,cur_index,group_name):
        group_index=[]
        for indices in cur_index:
            cur=map(int,indices[group_name])
            group_index.append(cur)
        return group_index


        
