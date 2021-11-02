import sys
from transformers import BertModel,BertTokenizer
from CR_walker import ProRec
from evaluation import evaluate_rec_gorecdial,evaluate_gen_gorecdial
from conf import add_generic_args,args

sys.path.append("..")
from data.gorecdial import GoRecDial
import torch
import torch.nn as nn
import os.path as osp
import json

import torch.nn.functional as F
from torch.autograd import Variable
import argparse

from torch_geometric.data import DataLoader



device_str = 'cuda:0'
device = torch.device(device_str)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='gorecdial_reason_128', type=str,
                    help="model name for saving stats and parameters")
parser.add_argument("--option",choices=['train','test','test_gen'],default='test_gen')
parser.add_argument("--pretrain",action='store_true')
parser.add_argument("--restore_best",action='store_true')
parser.add_argument("--graph_embed_size", type=int, default=128)
parser.add_argument("--utter_embed_size", type=int, default=128)
parser.add_argument("--negative_sample_ratio",type=int,default=3)
parser.add_argument("--train_epoch",type=int,default=60)
parser.add_argument("--pretrain_epoch",type=int,default=1)
parser.add_argument("--atten_hidden",type=int,default=20)
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--weight_decay",type=float,default=0.01)
parser.add_argument("--eval_batch",type=int,default=500)
parser.add_argument("--word_net",action='store_true')

t_args = parser.parse_args()


option=t_args.option
model_name=t_args.model_name




root=osp.dirname(osp.dirname(osp.abspath(__file__)))
save_path=osp.join(root,"saved","best_model_"+t_args.model_name+".pt")
path=osp.join(root,"data","gorecdial")
gorecdial_train=GoRecDial(path,flag="train")
gorecdial_test=GoRecDial(path,flag="test")
gorecdial_graph=GoRecDial(path,flag="graph")
gorecdial_bow=GoRecDial(path,flag="bow")
graph_data=gorecdial_graph[0]
bow_data=gorecdial_bow[0]


train_loader=DataLoader(gorecdial_train,batch_size=10,shuffle=True)
test_loader=DataLoader(gorecdial_test,batch_size=10,shuffle=False)


add_generic_args(dataset='gorecdial')





if option=="train":
    prorec=ProRec(device_str=device_str,graph_embed_size=t_args.graph_embed_size,utter_embed_size=t_args.utter_embed_size,dataset="gorecdial",word_net=t_args.word_net)
    if t_args.restore_best:
        print("restoring from best checkpoint...")
        state_dict=torch.load(save_path)
        prorec.load_state_dict(state_dict,strict=False)
        f=open('stats_'+model_name+'.json')
        stats_all=json.load(f)

        
        for i in range(len(stats_all['chat_1_ex'])):
            if stats_all['chat_1_ex'][i]>best_chat_1:
                best_chat_1=stats_all['chat_1_ex'][i]
            if stats_all['turn_1_ex'][i]>best_turn_1:
                best_turn_1=stats_all['turn_1_ex'][i]

        print("cur best chat 1:",best_chat_1)
        print("cur best turn 1:",best_turn_1)
    else:
        stats_all={"intent_accuracy":[],"turn_1":[],"turn_3":[],"chat_1":[],"chat_3":[],"turn_1_ex":[],"turn_3_ex":[],"chat_1_ex":[],"chat_3_ex":[]}
        best_chat_1=0
        best_turn_1=0

    
    
    unfreeze_layers = ["utter_embedder.rnn","graph_embedder","intent_selector","graph_walker","explicit_recommender","Wa","Ww"]

    for name ,param in prorec.named_parameters():
        param.requires_grad = False
        #print(name)
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                print(name) 
                break
    
    optimizer = torch.optim.Adam(prorec.parameters(), lr=t_args.lr,weight_decay=t_args.weight_decay)
    prorec.to(device)

    batch=0
    num=0

    max_epoch=t_args.train_epoch
    pretrain_epoch=t_args.pretrain_epoch


    
    if t_args.pretrain:
        num_pretrain=0
        for i in range(pretrain_epoch):
            for batch in train_loader:
                optimizer.zero_grad()
                tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label,alignment_index_word,alignment_batch_index_word,alignment_label_word=prorec.prepare_pretrain(batch.new_mention,batch.dialog_history,batch.intent,graph_data.edge_type,graph_data.edge_index,batch.rec_cand)
                loss=prorec.forward_pretrain(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label,alignment_index_word,alignment_batch_index_word,alignment_label_word)
                loss.backward()
                optimizer.step()
                print("pretrain iter ",num_pretrain,":",loss.item())
                num_pretrain+=1

   

    for i in range(max_epoch):
        for batch in train_loader:
            #print(batch.my_id)
            optimizer.zero_grad()
            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2,score_masks,word_index,word_batch_index=prorec.prepare_data_gorecdial(batch.dialog_history,batch.mention_history,batch.intent,batch.node_candidate1,batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,batch.label_1,batch.label_2,args['attribute_dict'],batch.rec_cand,bow_data.bow_embed)

            alignment_index,alignment_batch_index,alignment_label,alignment_index_word,alignment_batch_index_word,alignment_label_word=prorec.prepare_reg(batch.new_mention,batch.dialog_history,batch.intent,batch.rec_cand)

            intent,paths,loss=prorec.forward_gorecdial(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2,score_masks,alignment_index,alignment_batch_index,alignment_label,word_index,word_batch_index,alignment_index_word,alignment_batch_index_word,alignment_label_word)
            loss.backward()
            optimizer.step()

            print("iter ",num,":",loss.item())
            
            
            if (num+1) % t_args.eval_batch ==0:
                prorec.eval()
                intent_accuracy,turn_1,turn_3,chat_1,chat_3,turn_1_ex,turn_3_ex,chat_1_ex,chat_3_ex=evaluate_rec_gorecdial(test_loader,prorec,graph_data,bow_data,args)
                stats_all['intent_accuracy'].append(intent_accuracy)
                stats_all['turn_1'].append(turn_1)
                stats_all['turn_3'].append(turn_3)
                stats_all['chat_1'].append(chat_1)
                stats_all['chat_3'].append(chat_3)
                stats_all['turn_1_ex'].append(turn_1_ex)
                stats_all['turn_3_ex'].append(turn_3_ex)
                stats_all['chat_1_ex'].append(chat_1_ex)
                stats_all['chat_3_ex'].append(chat_3_ex)


                if turn_1_ex>best_turn_1 or chat_1_ex>best_chat_1:
                    if turn_1_ex>best_turn_1:
                        best_turn_1=turn_1_ex
                    if chat_1_ex>best_chat_1:
                        best_chat_1=chat_1_ex
                    print("saving model...")
                    torch.save(prorec.state_dict(),save_path)
            

                prorec.train()
                f=open('stats_'+model_name+'.json','w')

                
                json.dump(stats_all,f)
                f.close()
            num+=1

elif option=="test":
    print("testing model recommendation...")
    state_dict=torch.load(save_path,map_location=device_str)

    for key in state_dict.keys():
        print(key)
    prorec=ProRec(device_str=device_str,graph_embed_size=t_args.graph_embed_size,utter_embed_size=t_args.utter_embed_size,dataset="gorecdial")
    prorec.load_state_dict(state_dict,strict=False)
    prorec.eval()
    prorec.to(device)
    evaluate_rec_gorecdial(test_loader,prorec,graph_data,bow_data,args)


elif option=="test_gen":
    print("testing model generation...")
    state_dict=torch.load(save_path,map_location=device_str)

    for key in state_dict.keys():
        print(key)
    prorec=ProRec(device_str=device_str,graph_embed_size=t_args.graph_embed_size,utter_embed_size=t_args.utter_embed_size,negative_sample_ratio=t_args.negative_sample_ratio,word_net=t_args.word_net,dataset="gorecdial")
    prorec.load_state_dict(state_dict,strict=False)
    prorec.eval()
    prorec.to(device)
    evaluate_gen_gorecdial(test_loader,prorec,graph_data,bow_data,args,golden_intent=False)