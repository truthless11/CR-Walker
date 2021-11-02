import sys
from transformers import BertModel,BertTokenizer
from CR_walker import ProRec
from evaluation import evaluate_rec_redial,evaluate_gen_redial
from conf import add_generic_args,args
from entity_linker import match_nodes
from data.utils import da_tree_serial,utter_lexical_redial,utter_lexical_gorecdial
from copy import deepcopy

sys.path.append("..")
from data.utils import da_tree_serial
from data.redial import ReDial
import torch
import argparse
import torch.nn as nn
import os.path as osp
import json

import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange,tqdm
from torch_geometric.data import DataLoader


device_str = 'cuda:0'
device = torch.device(device_str)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='redial_reason_1best', type=str,
                    help="model name for saving stats and parameters")
parser.add_argument("--option",choices=['train','test','test_gen'],default='test_gen')
parser.add_argument("--pretrain",action='store_true')
parser.add_argument("--restore_best",action='store_true')
parser.add_argument("--graph_embed_size", type=int, default=128)
parser.add_argument("--utter_embed_size", type=int, default=128)
parser.add_argument("--negative_sample_ratio",type=int,default=5)
parser.add_argument("--train_epoch",type=int,default=60)
parser.add_argument("--pretrain_epoch",type=int,default=3)
parser.add_argument("--atten_hidden",type=int,default=20)
parser.add_argument("--lr",type=float,default=5e-4)
parser.add_argument("--weight_decay",type=float,default=0.01)
parser.add_argument("--eval_batch",type=int,default=5000)
parser.add_argument("--word_net",action='store_true')
t_args = parser.parse_args()



option=t_args.option
model_name=t_args.model_name

root=osp.dirname(osp.dirname(osp.abspath(__file__)))
save_path=osp.join(root,"saved","best_model_"+t_args.model_name+".pt")
save_path_1=osp.join(root,"saved","best_model_"+t_args.model_name+"_1.pt")
save_path_10=osp.join(root,"saved","best_model_"+t_args.model_name+"_10.pt")
save_path_50=osp.join(root,"saved","best_model_"+t_args.model_name+"_50.pt")
path=osp.join(root,"data","redial")
redial_train=ReDial(path,flag="train")
redial_test=ReDial(path,flag="test")
redial_graph=ReDial(path,flag="graph")
redial_rec=ReDial(path,flag="rec")
graph_data=redial_graph[0]


train_loader=DataLoader(redial_train,batch_size=20,shuffle=True)
test_loader=DataLoader(redial_test,batch_size=20,shuffle=False)

add_generic_args()



if t_args.option=="train":
    prorec=ProRec(device_str=device_str,graph_embed_size=t_args.graph_embed_size,utter_embed_size=t_args.utter_embed_size,negative_sample_ratio=t_args.negative_sample_ratio,atten_hidden=t_args.atten_hidden,word_net=t_args.word_net)
    if t_args.restore_best:
        print("restoring from best checkpoint...")
        state_dict=torch.load(save_path)
        prorec.load_state_dict(state_dict,strict=False)

        f=open('stats_'+model_name+'.json')
        stats_all=json.load(f)

        best_recall_1=0
        best_recall_10=0
        best_recall_50=0

        for i in range(len(stats_all['recall_1'])):
            if stats_all['recall_1'][i]>best_recall_1:
                best_recall_1=stats_all['recall_1'][i]
            if stats_all['recall_10'][i]>best_recall_10:
                best_recall_10=stats_all['recall_10'][i]
            if stats_all['recall_50'][i]>best_recall_50:
                best_recall_50=stats_all['recall_50'][i]
    else:
        best_recall_1=0
        best_recall_10=0
        best_recall_50=0
        stats_all={"recall_1":[],"recall_10":[],"recall_50":[]}

    unfreeze_layers = ["utter_embedder.rnn","intent_selector","graph_embedder","graph_walker","Wa","Ww"] #"intent_selector"

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
    num_pretrain=0
    
    pretrain_epoch=t_args.pretrain_epoch
    max_epoch=t_args.train_epoch


    if t_args.pretrain:
        for i in range(pretrain_epoch):
            for batch in train_loader:
                optimizer.zero_grad()
                tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label,alignment_index_word,alignment_batch_index_word,alignment_label_word=prorec.prepare_pretrain(batch.new_mention,batch.dialog_history,batch.intent,graph_data.edge_type,graph_data.edge_index)
                loss=prorec.forward_pretrain(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label,alignment_index_word,alignment_batch_index_word,alignment_label_word)
                loss.backward()
                optimizer.step()
                print("pretrain iter ",num_pretrain,":",loss.item())
                num_pretrain+=1
   

    for i in range(max_epoch):
        for batch in train_loader:
            optimizer.zero_grad()
            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,score_masks,word_index,word_batch_index=prorec.prepare_data_redial(batch.dialog_history,batch.mention_history,batch.intent,batch.node_candidate1,batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,batch.label_1,batch.label_2,batch.gold_pos,args['attribute_dict'],sample=True)
            alignment_index,alignment_batch_index,alignment_label,alignment_index_word,alignment_batch_index_word,alignment_label_word=prorec.prepare_reg(batch.new_mention,batch.dialog_history,batch.intent)


            intent,paths,loss=prorec.forward(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,score_masks,alignment_index,alignment_batch_index,alignment_label,word_index,word_batch_index,alignment_index_word,alignment_batch_index_word,alignment_label_word)

            loss.backward()
            optimizer.step()

            print("iter ",num,":",loss.item())
            
            if (num+1) % t_args.eval_batch == 0:
                prorec.eval()
                recall_1,recall_10,recall_50=evaluate_rec_redial(test_loader,prorec,graph_data,args)

                stats_all['recall_1'].append(recall_1)
                stats_all['recall_10'].append(recall_10)
                stats_all['recall_50'].append(recall_50)

                if recall_1>best_recall_1:
                    best_recall_1=recall_1
                    print("saving model...")
                    torch.save(prorec.state_dict(),save_path_1)
                    torch.save(prorec.state_dict(),save_path)
                if recall_10>best_recall_10:
                    best_recall_10=recall_10
                    print("saving model...")
                    torch.save(prorec.state_dict(),save_path_10)
                if recall_50>best_recall_50:
                    best_recall_50=recall_50
                    print("saving model...")
                    torch.save(prorec.state_dict(),save_path_50)
                prorec.train()
                f=open('stats_'+model_name+'.json','w')
                json.dump(stats_all,f)
                f.close()
            num+=1
    
elif t_args.option=="test":
    print("testing model recommendation...")
    state_dict=torch.load(save_path,map_location=device_str)

    for key in state_dict.keys():
        print(key)
    prorec=ProRec(device_str=device_str,graph_embed_size=t_args.graph_embed_size,utter_embed_size=t_args.utter_embed_size,negative_sample_ratio=t_args.negative_sample_ratio,word_net=t_args.word_net)
    prorec.load_state_dict(state_dict,strict=False)
    prorec.eval()
    prorec.to(device)
    evaluate_rec_redial(test_loader,prorec,graph_data,args)

elif t_args.option=="test_gen":
    print("testing model generation...")
    state_dict=torch.load(save_path,map_location=device_str)

    for key in state_dict.keys():
        print(key)
    prorec=ProRec(device_str=device_str,graph_embed_size=t_args.graph_embed_size,utter_embed_size=t_args.utter_embed_size,negative_sample_ratio=t_args.negative_sample_ratio,word_net=t_args.word_net)
    prorec.load_state_dict(state_dict,strict=False)
    prorec.eval()
    prorec.to(device)
    evaluate_gen_redial(test_loader,prorec,graph_data,args,golden_intent=False)