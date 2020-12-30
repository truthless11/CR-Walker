import sys
from transformers import BertModel,BertTokenizer
from CR_walker import ProRec
from evaluation import evaluate_gorecdial

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
#from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
#writer = SummaryWriter('runs/intent_classifier')
device_str = 'cuda:0'
device = torch.device(device_str)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='gorecdial_128', type=str,
                    help="model name for saving stats and parameters")
parser.add_argument("--option",choices=['train','test'],default='test')
parser.add_argument("--restore_best",action='store_true')
parser.add_argument("--graph_embed_size", type=int, default=128)
parser.add_argument("--utter_embed_size", type=int, default=128)
parser.add_argument("--negative_sample_ratio",type=int,default=3)
parser.add_argument("--train_epoch",type=int,default=30)
parser.add_argument("--atten_hidden",type=int,default=20)
parser.add_argument("--lr",type=float,default=5e-4)
parser.add_argument("--weight_decay",type=float,default=0.01)
parser.add_argument("--eval_batch",type=int,default=500)
args = parser.parse_args()


option=args.option
model_name=args.model_name




root=osp.dirname(osp.dirname(osp.abspath(__file__)))
save_path=osp.join(root,"saved","best_model_"+args.model_name+".pt")
path=osp.join(root,"data","gorecdial")
gorecdial_train=GoRecDial(path,flag="train")
gorecdial_test=GoRecDial(path,flag="test")
gorecdial_graph=GoRecDial(path,flag="graph")
gorecdial_bow=GoRecDial(path,flag="bow")
graph_data=gorecdial_graph[0]
bow_data=gorecdial_bow[0]



train_loader=DataLoader(gorecdial_train,batch_size=10,shuffle=True)
test_loader=DataLoader(gorecdial_test,batch_size=10,shuffle=False)





if option=="train":
    prorec=ProRec(device_str=device_str,graph_embed_size=args.graph_embed_size,utter_embed_size=args.utter_embed_size,dataset="gorecdial")
    if args.restore_best:
        print("restoring from best checkpoint...")
        state_dict=torch.load(save_path)
        prorec.load_state_dict(state_dict)
        f=open('stats_'+model_name+'.json')
        stats_all=json.load(f)
        # if turn_1_ex>best_turn_1 or chat_1_ex>best_chat_1:
        #             if turn_1_ex>best_turn_1:
        #                 best_turn_1=turn_1_ex
        #             if chat_1_ex>best_chat_1:
        #                 best_chat_1=chat_1_ex
        best_chat_1=0
        best_turn_1=0
        for i in range(len(stats_all['chat_1_ex'])):
            if stats_all['chat_1_ex'][i]>best_chat_1:
                best_chat_1=stats_all['chat_1_ex'][i]
            if stats_all['turn_1_ex'][i]>best_turn_1:
                best_turn_1=stats_all['turn_1_ex'][i]
    else:
        stats_all={"intent_accuracy":[],"turn_1":[],"turn_3":[],"chat_1":[],"chat_3":[],"turn_1_ex":[],"turn_3_ex":[],"chat_1_ex":[],"chat_3_ex":[]}
        best_chat_1=0
        best_turn_1=0

   
    #state_dict=torch.load(save_path)
   # prorec.load_state_dict(state_dict)
    unfreeze_layers = ["utter_embedder.rnn","graph_embedder","intent_selector","graph_walker","explicit_recommender"]

    for name ,param in prorec.named_parameters():
        param.requires_grad = False
        #print(name)
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                print(name) 
                break

    optimizer = torch.optim.Adam(prorec.parameters(), lr=1e-3,weight_decay=args.weight_decay)
    prorec.to(device)

    batch=0
    num=0


    #best_chat_1=0
    #best_turn_1=0

    max_epoch=10
   

    for i in range(max_epoch):
        for batch in train_loader:
            #print(batch.my_id)
            optimizer.zero_grad()
            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2=prorec.prepare_data_gorecdial(batch.dialog_history,batch.mention_history,batch.intent,batch.node_candidate1,batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,batch.label_c,batch.label_2,batch.rec_cand,bow_data.bow_embed)


            
            intent,paths,loss=prorec.forward_gorecdial(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2)
            loss.backward()
            optimizer.step()

            print("iter ",num,":",loss.item())
            
            if (num+1) % args.eval_batch ==0:
                prorec.eval() 
                intent_accuracy,turn_1,turn_3,chat_1,chat_3,turn_1_ex,turn_3_ex,chat_1_ex,chat_3_ex=evaluate_gorecdial(test_loader,prorec,graph_data,bow_data)
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
        #break
else:
    print("testing model...")
    state_dict=torch.load(save_path,map_location=device_str)

    for key in state_dict.keys():
        print(key)
    prorec=ProRec(device_str=device_str,graph_embed_size=args.graph_embed_size,utter_embed_size=args.utter_embed_size,dataset="gorecdial")
    prorec.load_state_dict(state_dict)
    prorec.eval()
    


    unfreeze_layers = ["utter_embedder.rnn","graph_embedder","intent_selector","graph_walker","explicit_recommender"]

    for name ,param in prorec.named_parameters():
        param.requires_grad = False
        #print(name)
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                print(name)
                break
    
    #prorec.load_state_dict(state_dict)
    #prorec.eval()
    prorec.to(device)
    intent_accuracy,turn_1,turn_3,chat_1,chat_3,turn_1_ex,turn_3_ex,chat_1_ex,chat_3_ex=evaluate_gorecdial(test_loader,prorec,graph_data,bow_data,gen_DA=True,gen_utter=True)


# correct=0
# accuracy=[]

# max_epoch=5
# running_loss=0

# stats=[0,0,0]

# all_intent=["chat","question","recommend"]
# for i in range(max_epoch):
#     for batch in train_loader:
#         #print(i," ",num)
#         #print(batch)
#         #print(batch.mention_history)
#         #print(batch.graph_size)
#         optimizer.zero_grad()
#         predicted=prorec.forward(batch.dialog_history,graph_data.node_feature,graph_data.edge_type,graph_data.edge_index,batch.mention_history)
#         #print(torch.softmax(predicted,dim=-1))
#         selected=torch.softmax(predicted,dim=-1).max(dim=-1).indices
#         #print("selected:",selected)
#         label=get_intent(batch.intent)
#         #print(label)
#         #print("label:",label)
#         loss = loss_fn(predicted,label)
#         #print(loss.item())
#         loss.backward()
#         optimizer.step()

#         print("iter ",num,":",loss.item())
#         running_loss=0

#         if (num+1) % 200 ==0:
#             correct=0
#             da_distrib=[0,0,0]
#             for j,test_batch in enumerate(test_loader,0):
#                 predicted=prorec.forward(test_batch.dialog_history,graph_data.node_feature,graph_data.edge_type,graph_data.edge_index,test_batch.mention_history)
#                 selected=torch.softmax(predicted,dim=-1).max(dim=-1).indices
#                 label=get_intent(test_batch.intent)
#                 for p in range(10):
#                     da_distrib[selected[p]]=da_distrib[selected[p]]+1
#                     if selected[p]==label[p]:
#                         correct+=1
#                 if j==50:
#                     break
#             print('da_distribution:',da_distrib)
#             print("test accuracy:",correct/500)
#             accuracy.append(correct/500)
#             f=open('stats_new.json','w')
#             json.dump(accuracy,f)
#             f.close()
#         num+=1