import sys
from transformers import BertModel,BertTokenizer
from CR_walker import ProRec
from evaluation import evaluate_redial,DA_interact
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
#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter('runs/intent_classifier')
device_str = 'cuda:0'
device = torch.device(device_str)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='redial_128', type=str,
                    help="model name for saving stats and parameters")
parser.add_argument("--option",choices=['train','test','interact'],default='test')
parser.add_argument("--pretrain",action='store_true')
parser.add_argument("--restore_best",action='store_true')
parser.add_argument("--graph_embed_size", type=int, default=128)
parser.add_argument("--utter_embed_size", type=int, default=128)
parser.add_argument("--negative_sample_ratio",type=int,default=3)
parser.add_argument("--train_epoch",type=int,default=30)
parser.add_argument("--pretrain_epoch",type=int,default=3)
parser.add_argument("--atten_hidden",type=int,default=20)
parser.add_argument("--lr",type=float,default=5e-4)
parser.add_argument("--weight_decay",type=float,default=0.01)
parser.add_argument("--eval_batch",type=int,default=600)
args = parser.parse_args()



option=args.option
model_name=args.model_name

root=osp.dirname(osp.dirname(osp.abspath(__file__)))
save_path=osp.join(root,"saved","best_model_"+args.model_name+".pt")
path=osp.join(root,"data","redial")
redial_train=ReDial(path,flag="train")
redial_test=ReDial(path,flag="test")
redial_graph=ReDial(path,flag="graph")
redial_rec=ReDial(path,flag="rec")
graph_data=redial_graph[0]


train_loader=DataLoader(redial_train,batch_size=20,shuffle=True)
test_loader=DataLoader(redial_test,batch_size=20,shuffle=False)



if args.option=="train":
    prorec=ProRec(device_str=device_str,graph_embed_size=args.graph_embed_size,utter_embed_size=args.utter_embed_size,negative_sample_ratio=args.negative_sample_ratio,atten_hidden=args.atten_hidden)
    if args.restore_best:
        print("restoring from best checkpoint...")
        state_dict=torch.load(save_path)
        prorec.load_state_dict(state_dict)
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
                best_recall_50=stats_all['recall_1'][i]
    else:
        best_recall_1=0
        best_recall_10=0
        best_recall_50=0
        stats_all={"recall_1":[],"recall_10":[],"recall_50":[]}

    unfreeze_layers = ["utter_embedder.rnn","intent_selector","graph_embedder","graph_walker"] #"intent_selector"

    for name ,param in prorec.named_parameters():
        param.requires_grad = False
        #print(name)
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                print(name) 
                break
    

    optimizer = torch.optim.Adam(prorec.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    prorec.to(device)

    batch=0
    num=0
    num_pretrain=0
    
    pretrain_epoch=args.pretrain_epoch
    max_epoch=args.train_epoch


    if args.pretrain:
        for i in range(pretrain_epoch):
            for batch in train_loader:
                optimizer.zero_grad()
                tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label=prorec.prepare_pretrain(batch.new_mention,batch.intent,batch.dialog_history,graph_data.edge_type,graph_data.edge_index)
                #tokenized_dialog,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label=prorec.prepare_pretrain(batch.mention_history,batch.intent,batch.dialog_history,graph_data.edge_type,graph_data.edge_index)
                #print(alignment_label)
                loss=prorec.forward_pretrain(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label,intent_label)
                loss.backward()
                optimizer.step()
                print("pretrain iter ",num_pretrain,":",loss.item())
                num_pretrain+=1

    
   

    for i in range(max_epoch):
        for batch in train_loader:
            #print(batch.my_id)
            optimizer.zero_grad()
            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2=prorec.prepare_data_redial(batch.dialog_history,batch.mention_history,batch.intent,batch.node_candidate1,batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,batch.label_1,batch.label_2,sample=True)
            alignment_index,alignment_batch_index,alignment_label=prorec.prepare_pretrain(batch.new_mention)
            #alignment_index,alignment_batch_index,alignment_label=prorec.prepare_pretrain(batch.mention_history)
            
            intent,paths,loss=prorec.forward(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,alignment_index,alignment_batch_index,alignment_label)

            #loss_re=prorec.forward_pretrain(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,alignment_index,alignment_batch_index,alignment_label)

            #tot_loss=pretrain_lambda*loss_re+loss
            loss.backward()
            optimizer.step()

            print("iter ",num,":",loss.item())
            
            if (num+1) % args.eval_batch == 0:
                prorec.eval()
                recall_1,recall_10,recall_50=evaluate_redial(test_loader,prorec,graph_data)

                stats_all['recall_1'].append(recall_1)
                stats_all['recall_10'].append(recall_10)
                stats_all['recall_50'].append(recall_50)

                # if recall_1>best_recall_1 or recall_10>best_recall_10 or recall_50>best_recall_50:
                if recall_1>best_recall_1:
                    best_recall_1=recall_1
                if recall_10>best_recall_10:
                    best_recall_10=recall_10
                if recall_50>best_recall_50:
                    best_recall_50=recall_50
                    print("saving model...")
                    torch.save(prorec.state_dict(),save_path)
                #     print("saving model...")
                #     torch.save(prorec.state_dict(),save_path)
                prorec.train()
                f=open('stats_'+model_name+'.json','w')
                json.dump(stats_all,f)
                f.close()
            num+=1
        #break
    
elif args.option=="test":
    #print("testing model...")
    state_dict=torch.load(save_path,map_location=device_str)

    for key in state_dict.keys():
        print(key)
    prorec=ProRec(device_str=device_str,graph_embed_size=args.graph_embed_size,utter_embed_size=args.utter_embed_size,negative_sample_ratio=args.negative_sample_ratio)
    prorec.load_state_dict(state_dict,strict=False)
    prorec.eval()
    # unfreeze_layers = ["utter_embedder.rnn","graph_embedder.gcn","intent_selector","graph_walker"]

    # for name ,param in prorec.named_parameters():
    #     param.requires_grad = False
    #     #print(name)
    #     for ele in unfreeze_layers:
    #         if ele in name:
    #             param.requires_grad = True
    #             print(name)
    #             break
    
    #prorec.load_state_dict(state_dict)
    #prorec.eval()
    prorec.to(device)
    evaluate_redial(test_loader,prorec,graph_data,gen_DA=True,gen_utter=True,golden_intent=False)
else:
    from generator import Generator
    from conf import args,add_generic_args
    add_generic_args()
    gener=Generator(args['gen_conf'])
    state_dict=torch.load(save_path,map_location=device_str)

    for key in state_dict.keys():
        print(key)
    prorec=ProRec(device_str=device_str,graph_embed_size=args.graph_embed_size,utter_embed_size=args.utter_embed_size,negative_sample_ratio=args.negative_sample_ratio)
    prorec.load_state_dict(state_dict,strict=False)
    prorec.eval()
    prorec.to(device)

    with open('/home/mawenchang/PROREC-Torch/data/id2name_redial.json', 'r') as f:
        id2name = json.load(f)
    #with open('/home/mawenchang/PROREC-Torch/data/mid2name_redial.json', 'r') as f:
    with open('/home/mawenchang/PROREC-Torch/data/mid2name_redial.json', 'r') as f:
        mid2name = json.load(f)
    dialog_history=[]
    mentioned=[]
    memory_turns=2
    mentioned_seq=[[] for _ in range(memory_turns)]
    #mention_level
    while(True):
        utter=input()
        dialog_history.append(utter)
        
        matched=match_nodes(utter,mentioned)
        #print(matched)
        for i in range(memory_turns-1):
            mentioned_seq[i]=deepcopy(mentioned_seq[i+1])
        for n in matched:
            mentioned_seq[-1].append(n)

        mentioned=[]
        for item in mentioned_seq:
            mentioned+=item
        mentioned=list(set(mentioned))
        print(mentioned)

        da=DA_interact(prorec,[dialog_history],[mentioned],graph_data)
        DA=da_tree_serial(da,id2name)
        context=utter.lower()
        gpt_in=context+" @ "+DA+" &"
        #gpt_in=context+"@"+item['intent']+" &" 
        #print(gpt_in.lower())
        generated=gener.generate(gpt_in.lower())
        print(generated)
        #len1=len(da['layer1'])
        dialog_history.append(generated)
        for i in range(memory_turns-1):
            mentioned_seq[i]=deepcopy(mentioned_seq[i+1])
        for nod in da['layer1']:
            mentioned_seq[-1].append(nod)
        for item in da['layer2']:
            for nod in item:
                mentioned_seq[-1].append(nod)
        # for n in matched:
        #     mentioned_seq[-1].append(n)


        

    
    #for batch in test_loader:
    
    #print(batch.mention_history[0][-1])
    #print(batch.dialog_history)
    #
        


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