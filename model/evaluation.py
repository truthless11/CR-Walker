import json
import numpy as np
import torch
import sys
import os.path as osp
from tqdm import tqdm
from termcolor import colored
from copy import deepcopy
from conf import args,add_generic_args
import random

from CR_walker import ProRec
from torch_geometric.data import DataLoader

sys.path.append("..")
from data.redial import ReDial
from data.gorecdial import GoRecDial
from data.metrics import bleu,f1_score,distinct_n_grams
from data.utils import da_tree_serial,utter_lexical_redial,utter_lexical_gorecdial

def select_intent(sel_intent,mentioned,cand_movs=None):
    device=torch.device('cuda:0')
    last=[]
    bat=[]  
    sel=[]
    grp=[]
    #itt=[]
    grp_bat=[]
    node_candidate1=[]
    cur_grp_size=sel_intent.size()[0]
    for i in range(cur_grp_size):
        cand=[]
        last.append(args['none_node'])
        grp_bat.append(i)
        #itt.append(sel_intent[i])
        if sel_intent[i]==0:#chat
            for item in mentioned[i]:
                cand.append(item)
                sel.append(item)
                bat.append(i)
                grp.append(i)
            if len(mentioned[i])==0:
                cand.append(args['none_node'])
                sel.append(args['none_node'])
                bat.append(i)
                grp.append(i)
        elif sel_intent[i]==1:#question
            for item in args['generals']:
                cand.append(item)
                sel.append(item)
                bat.append(i)
                grp.append(i)
        else: #recommend
            if args['dataset']=='redial':
                cand.append(-1)
                for item in range(args['movie_count']):
                    sel.append(item)
                    bat.append(i)
                    grp.append(i)
            else:
                for item in range(5):
                    cand.append(cand_movs[i][item])
                    sel.append(cand_movs[i][item])
                    bat.append(i)
                    grp.append(i)
        node_candidate1.append(cand)
    sel_index=torch.Tensor(sel).long().to(device=device)
    grp_index=torch.Tensor(grp).long().to(device=device)
    batch_index=torch.Tensor(bat).long().to(device=device)
    intent_index=sel_intent
    #print(intent_index)
    grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
    last_index=torch.Tensor(last).long().to(device=device)
    return sel_index,grp_index,batch_index,grp_bat_index,last_index,node_candidate1


def select_layer_1(nodes,step,step_grp,intent_label,node_candidate1,label_1,mentioned,filter=False):
    device=torch.device('cuda:0')
    cur_grp_size=intent_label.size()[0]
    cand_num=[0 for _ in range(cur_grp_size+1)]
    all_selected=[]
    for i in step_grp:
        cand_num[i]+=1
    start=0
    end=cand_num[0]
    last=[]
    bat=[]  
    sel=[]
    grp=[]
    itt=[]
    grp_bat=[]
    grp_num=0
    all_candidates=[]
    for num in range(cur_grp_size):
        selected=[]
        score=step[start:end].tolist()
        if len(node_candidate1[num])!=0:
            best_idx=score.index(max(score))
            if intent_label[num]==0:
                for nod,scr in enumerate(score):
                    if scr>args['threshold'][0][0]:
                        last.append(node_candidate1[num][nod])
                        selected.append(node_candidate1[num][nod])
                        if nodes[node_candidate1[num][nod]]['type']=='Movie':
                            node_candidate2=list(args['attribute_dict'][node_candidate1[num][nod]])+[args['none_node']]
                        else:
                            node_candidate2=[args['none_node']]
                        all_candidates.append(node_candidate2)
                        sel=sel+node_candidate2
                        bat=bat+[num for _ in range(len(node_candidate2))]
                        grp=grp+[grp_num for _ in range(len(node_candidate2))]
                        grp_bat.append(num)
                        itt.append(intent_label[num])
                        grp_num+=1
                        #print(colored(nodes[test_batch.node_candidate1[num][nod]]['name'],"blue")," ",scr)
            elif intent_label[num]==1:
                step_general_dict=deepcopy(args['generals_dict'])
                for n in mentioned[num]:
                    if nodes[n]['type']!="Attr" and nodes[n]['type']!="None":
                        if nodes[n]['type']=="Person":
                            for rol in nodes[n]['role']:
                                step_general_dict[rol].add(nodes[n]['global'])
                        else:
                            step_general_dict[nodes[n]['type']].add(nodes[n]['global'])
                        if nodes[n]['type']=="Movie":
                            for attr in args['attribute_dict'][nodes[n]['global']]:
                                if nodes[attr]['type']!="Attr":
                                    if nodes[attr]['type']=="Person":
                                        for rol in nodes[attr]['role']:
                                            step_general_dict[rol].add(attr)
                                    else:
                                        step_general_dict[nodes[attr]['type']].add(attr)
                for nod,scr in enumerate(score):
                    if scr>args['threshold'][0][1]:
                        selected.append(node_candidate1[num][nod])
                        last.append(node_candidate1[num][nod])
                        cur_gen=nodes[node_candidate1[num][nod]]['name']
                        node_candidate2=list(step_general_dict[cur_gen])+[args['none_node']]
                        all_candidates.append(node_candidate2)
                        sel=sel+node_candidate2
                        bat=bat+[num for _ in range(len(node_candidate2))]
                        grp=grp+[grp_num for _ in range(len(node_candidate2))]
                        grp_bat.append(num)
                        itt.append(intent_label[num])
                        grp_num+=1
                        #print(colored(nodes[test_batch.node_candidate1[num][nod]]['name'],"blue")," ",scr)
            else:

                if filter==True:
                    #filter all previously mentioned movies 
                    for n in range(args['movie_count']):
                        if n in mentioned[num]:
                            score[n]=score[n]-1e3
                if args['sample']!=-1:
                    #sample top 1~3 movie candidates
                    score_torch=torch.Tensor(score)
                    top_k=score_torch.topk(5).indices.numpy()
                    num_sample=random.sample(range(1,args['sample']+1),1)[0]
                    #sampler=random.sample(range(5),num_sample)
                    for item in range(num_sample):
                        if args['dataset']=='redial':
                            movie_idx=top_k[item]
                        else:
                            movie_idx=node_candidate1[num][item]
                        selected.append(movie_idx)
                        last.append(movie_idx)
                        node_candidate2=list(args['attribute_dict'][movie_idx])+[args['none_node']]
                        all_candidates.append(node_candidate2)
                        sel=sel+node_candidate2
                        bat=bat+[num for _ in range(len(node_candidate2))]
                        grp=grp+[grp_num for _ in range(len(node_candidate2))]
                        grp_bat.append(num)
                        itt.append(intent_label[num])
                        grp_num+=1
                else:
                    for nod,scr in enumerate(score):
                        if scr>args['threshold'][0][2]:
                            selected_num+=1
                            selected.append(nod)
                            last.append(nod)
                            node_candidate2=list(args['attribute_dict'][nod])+[args['none_node']]
                            all_candidates.append(node_candidate2)
                            sel=sel+node_candidate2
                            bat=bat+[num for _ in range(len(node_candidate2))]
                            grp=grp+[grp_num for _ in range(len(node_candidate2))]
                            grp_bat.append(num)
                            itt.append(intent_label[num])
                            grp_num+=1
        all_selected.append(selected)
        start+=cand_num[num]
        end+=cand_num[num+1]
    sel_index=torch.Tensor(sel).long().to(device=device)
    grp_index=torch.Tensor(grp).long().to(device=device)
    batch_index=torch.Tensor(bat).long().to(device=device)
    intent_index=torch.Tensor(itt).long().to(device=device)
    grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
    last_index=torch.Tensor(last).long().to(device=device)
    return sel_index,grp_index,batch_index,intent_index,grp_bat_index,last_index,all_candidates,all_selected


def select_layer_2(step,step_grp,grp_batch,intent_label,node_candidate2,batch_size):
    #device=torch.device('cuda:0')
    print(intent_label)
    cur_grp_size=step_grp[-1]+1
    cand_num=[0 for _ in range(cur_grp_size+1)]
    for i in step_grp:
        cand_num[i]+=1
    start=0
    end=cand_num[0]
    selected=[[] for _ in range(batch_size)]
    #print("grp_size",cur_grp_size)
    #print(grp_batch)
    #print(step_grp)
    for num in range(cur_grp_size):
        score=step[start:end].tolist()
        threshold=args['threshold'][1][intent_label[num]]
        k=0
        cur_sel=[]
        for i,scr in enumerate(score):
            if scr>threshold and i!=len(score)-1:
                cur_sel.append(node_candidate2[num][i])
                k+=1
            if k==args['max_leaf']:
                break
        selected[grp_batch[num]].append(cur_sel)
        #index=score.index(max(score))
        #selected[grp_batch[num]].append([node_candidate2[num][index]])
        start+=cand_num[num]
        end+=cand_num[num+1]

    return selected

        
def DA_interact(model:ProRec,dialog_history,mention_history,graph_data):
    with torch.no_grad():
        tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index=model.prepare_data_interactive(dialog_history,mention_history,graph_data.edge_type,graph_data.edge_index)
        intent=model.get_intent(tokenized_dialog,all_length,maxlen,init_hidden)
        selected=intent.max(dim=-1).indices
        sel_index_1i,grp_index_1i,batch_index_1i,grp_bat_index_1i,last_index_1i,node_candidate1i=select_intent(selected,mention_history)
        cur_batch_size=intent.size()[0]

        step1i=model.inference_redial(selected,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_1i,batch_index_1i,grp_index_1i,grp_bat_index_1i,last_index_1i)
        step1i=step1i.cpu().numpy()
        step_grp=grp_index_1i.cpu().numpy()
        # for i in step_grp:
        #     cand_num[i]+=1
        sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,node_candidate2,selected_1=select_layer_1(args['nodes'],step1i,step_grp,selected,node_candidate1i,[],mention_history,filter=True)
        #print("sel_grp_index:",grp_index_2)
        if grp_index_2.size()[0]!=0:
            step2=model.inference_redial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,layer=1)
            selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size)
        else:
            selected_2=[[[]]]

        
        all_intent=['chat','question','recommend']

        
        for i in range(len(selected_1)):
            for j in range(len(selected_1[i])):
                selected_1[i][j]=int(selected_1[i][j])
        for i in range(len(selected_2)):
            for j in range(len(selected_2[i])):
                for k in range(len(selected_2[i][j])):
                    selected_2[i][j][k]=int(selected_2[i][j][k])
        print(selected_1)
        print(selected_2)
        # input()
        data={'intent':all_intent[selected[0]],'layer1':selected_1[0],'layer2':selected_2[0]}
    return data


def evaluate_redial(test_loader:DataLoader, model:ProRec,graph_data, eval_batch=None,golden_intent=True,gen_DA=False,gen_utter=True):

    add_generic_args(dataset='redial')

    recall_1=0
    recall_10=0
    recall_50=0
    intent_accuracy=0
    da_distrib=[0,0,0]
    tot_rec=0
    tot=0
    

    batches=0
    model.eval()

    generated_DAs=[]
    with torch.no_grad():
        for test_batch in tqdm(test_loader):
            #print("test_iter:",str(j))

            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2=model.prepare_data_redial(test_batch.dialog_history,test_batch.mention_history,test_batch.intent,test_batch.node_candidate1,test_batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,test_batch.label_1,test_batch.label_2)

            #print(test_batch.rec_cand)
            #print(intent_indices[0])

            intent=model.get_intent(tokenized_dialog,all_length,maxlen,init_hidden)
            selected=intent.max(dim=-1).indices
            step1=model.inference_redial(intent_indices[0],tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices[0],sel_batch_indices[0],sel_group_indices[0],grp_batch_indices[0],last_indices[0])
            # rec=model.get_rec(tokenized_dialog,edge_type,edge_index,node_feature,)
            cur_batch_size=intent.size()[0]


            if gen_DA:
                if golden_intent:
                    #cand_num=[0 for _ in range(cur_batch_size+1)]
                    step1=step1.cpu().numpy()
                    step_grp=sel_group_indices[0].cpu().numpy()
                    # for i in step_grp:
                    #     cand_num[i]+=1
                    sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,node_candidate2,selected_1=select_layer_1(args['nodes'],step1,step_grp,intent_label,test_batch.node_candidate1,test_batch.label_1,test_batch.mention_history)
                    step2=model.inference_redial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,layer=1)

                    selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size)
                    all_intent=['chat','question','recommend']

                    
                    for i in range(len(selected_1)):
                        for j in range(len(selected_1[i])):
                            selected_1[i][j]=int(selected_1[i][j])
                    for i in range(len(selected_2)):
                        for j in range(len(selected_2[i])):
                            for k in range(len(selected_2[i][j])):
                                selected_2[i][j][k]=int(selected_2[i][j][k])
                    #print(selected_1)
                    #print(selected_2)
                    #input()

                    for num in range(cur_batch_size):
                        data={'intent':all_intent[intent_label[num]],'layer1':selected_1[num],'layer2':selected_2[num],'key':test_batch.my_id[num]}
                        generated_DAs.append(data)
                else:
                    sel_index_1i,grp_index_1i,batch_index_1i,grp_bat_index_1i,last_index_1i,node_candidate1i=select_intent(selected,test_batch.mention_history)
                    step1i=model.inference_redial(selected,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_1i,batch_index_1i,grp_index_1i,grp_bat_index_1i,last_index_1i)
                    step1i=step1i.cpu().numpy()
                    step_grp=grp_index_1i.cpu().numpy()
                    # for i in step_grp:
                    #     cand_num[i]+=1
                    sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,node_candidate2,selected_1=select_layer_1(args['nodes'],step1i,step_grp,selected,node_candidate1i,test_batch.label_1,test_batch.mention_history)
                    #print("sel_grp_index:",grp_index_2)
                    if grp_index_2.size()[0]!=0:
                        step2=model.inference_redial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,layer=1)
                        selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size)
                    else:
                        selected_2=[[[]]]

                    
                    all_intent=['chat','question','recommend']

                    
                    for i in range(len(selected_1)):
                        for j in range(len(selected_1[i])):
                            selected_1[i][j]=int(selected_1[i][j])
                    for i in range(len(selected_2)):
                        for j in range(len(selected_2[i])):
                            for k in range(len(selected_2[i][j])):
                                selected_2[i][j][k]=int(selected_2[i][j][k])
                    print(selected_1)
                    print(selected_2)
                    #input()

                    for num in range(cur_batch_size):
                        data={'intent':all_intent[selected[num]],'layer1':selected_1[num],'layer2':selected_2[num],'key':test_batch.my_id[num]}
                        generated_DAs.append(data)



            cand_num=[0 for _ in range(cur_batch_size+1)]
            step_batch=sel_batch_indices[0].cpu().numpy()
            for i in step_batch:
                cand_num[i]+=1
            # candidate=candidates[0]
            # print(candidate)
            # print(step.size())

            start=0
            end=cand_num[0]
            dialog_act=[]
            for num in range(cur_batch_size):
                tot+=1
                my_label=test_batch.label_1[num]
                score=step1[start:end].tolist()
                if intent_label[num]==2:
                    wrong_count=[0 for _ in range(len(my_label))]
                    for item in score:
                        for p,idx in enumerate(my_label):
                            if item>=score[idx]:
                                wrong_count[p]+=1
                    
                    for item in wrong_count:
                        tot_rec+=1
                        if item<2:
                            recall_1+=1
                        if item<11:
                            recall_10+=1
                        if item<51:
                            recall_50+=1
                
                start+=cand_num[num]
                end+=cand_num[num+1]
            
                da_distrib[selected[num]]=da_distrib[selected[num]]+1
                if selected[num]==intent_label[num]:
                    intent_accuracy+=1
            #print(intent_accuracy/tot)
            if batches==eval_batch:
                break


            batches+=1


        recall_1=recall_1/tot_rec
        recall_10=recall_10/tot_rec
        recall_50=recall_50/tot_rec
        intent_accuracy=intent_accuracy/tot

        print("tot_rec",tot_rec)
        print("recall_1",recall_1)
        print('recall_10:',recall_10)
        print("recall_50:",recall_50)
        print("intent_accuracy:",intent_accuracy)
        if gen_DA==True:
            f=open(args['DA_save_path'],'w')
            json.dump(generated_DAs,f)

        if gen_utter==True:
            from generator import Generator

            gener=Generator(args['gen_conf'])

            with open(args['DA_save_path'],'r') as f:
                generated_DAs=json.load(f)
            
            generate_all={}

            dataset=ReDial(args['data_path'],flag="test")
            for i,item in tqdm(enumerate(generated_DAs)):
                DA=da_tree_serial(item,args['id2name'])
                if len(dataset[i].dialog_history)!=0:
                    context=dataset[i].dialog_history[-1]
                else:
                    context="hello"
                
                context=utter_lexical_redial(context,args['mid2name'])
                gpt_in=context+" @ "+DA+" &"

                print(gpt_in.lower())
                generated=gener.generate(gpt_in.lower())
                print(utter_lexical_redial(dataset[i].oracle_response,args['mid2name']))
                cur_turn={"generated":generated,"label":utter_lexical_redial(dataset[i].oracle_response,args['mid2name'])}
                generate_all[item['key']]=cur_turn
                if (i+1) % 50 ==0:
                    print("saving results...")
                    with open(args['utter_save_path'],'w') as f:
                        json.dump(generate_all,f)
            print("saving results...")
            with open(args['utter_save_path'],'w') as f:
                json.dump(generate_all,f)
            
            lines = [item['generated'].strip() for item in generate_all.values()]
            bleu_array = []
            f1_array = []
            
            k=0
            for item in generate_all.values():
                k+=1
                ground, gen = [item['label']], item['generated']
                bleu_array.append(bleu(gen, ground))
                f1_array.append(f1_score(gen, ground))
            print("BLEU:",np.mean(bleu_array))
            print("F1:",np.mean(f1_array))


            tokenized = [line.split() for line in lines]
            for n in range(1, 6):
                cnt, percent = distinct_n_grams(tokenized, n)
                print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')

    return recall_1,recall_10,recall_50


def evaluate_gorecdial(test_loader:DataLoader, model:ProRec,graph_data, bow_data, batch_size=10,eval_batch=None,gen_DA=False,gen_utter=False):
    add_generic_args(dataset='gorecdial')
    
    correct=0
    da_distrib=[0,0,0]
    

    intent_accuracy=0

    tot_turns=0
    tot_dialog=0

    turn_1=0
    turn_3=0
    chat_1=0
    chat_3=0

    turn_1_ex=0
    turn_3_ex=0
    chat_1_ex=0
    chat_3_ex=0

    splits=6
    accuracy_split=[0 for _ in range(splits)]
    tot_split=[0 for _ in range(splits)]
    correct=[]
    turn=0
    generated_DAs=[]

    for test_batch in tqdm(test_loader):
        tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2 = model.prepare_data_gorecdial(test_batch.dialog_history,test_batch.mention_history,test_batch.intent,test_batch.node_candidate1,test_batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,test_batch.label_c,test_batch.label_2,test_batch.rec_cand,bow_data.bow_embed)
        intent=model.get_intent(tokenized_dialog,all_length,maxlen,init_hidden)

        #print(test_batch.rec_cand)
        rec_index,rec_batch_index, rec_group_index,rec_grp_bat_index,rec_last_index,rec_intent,golden=model.prepare_rectest(test_batch.rec_cand)

        step,rec=model.inference_gorecdial(rec_intent,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,rec_index,rec_batch_index,rec_group_index,rec_grp_bat_index,rec_last_index,bow_embed)

        selected=torch.softmax(intent,dim=-1).max(dim=-1).indices
        cur_batch_size=intent.size()[0]
        for num in range(cur_batch_size):
            da_distrib[selected[num]]=da_distrib[selected[num]]+1
            if selected[num]==intent_label[num]:
                intent_accuracy+=1
        # rec=model.get_rec(tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,node_feature,)

        #score=score.view(-1,5)
        step=step.view(-1,5).cpu().detach().numpy()
        mention=mention_index.cpu().tolist()
        mentioned_idx=mention_batch_index.cpu().tolist()
        mentioned=[[] for _ in range(cur_batch_size)]
        for i,item in enumerate(mention_batch_index):
            mentioned[item].append(mention[i])



        if gen_DA==True:
            sel_index_1i,grp_index_1i,batch_index_1i,grp_bat_index_1i,last_index_1i,node_candidate1i=select_intent(selected,mentioned,test_batch.rec_cand)
            step1i=model.inference_gorecdial(selected,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_1i,batch_index_1i,grp_index_1i,grp_bat_index_1i,last_index_1i,layer=0)
            step1i=step1i.cpu().detach().numpy()
            step_grp=grp_index_1i.cpu().numpy()
            # for i in step_grp:
            #     cand_num[i]+=1
            sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,node_candidate2,selected_1=select_layer_1(args['nodes'],step1i,step_grp,selected,node_candidate1i,test_batch.label_1,mentioned)
            
            step2=model.inference_redial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,layer=1)

            selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size)
            all_intent=['chat','question','recommend']

            
            for i in range(len(selected_1)):
                for j in range(len(selected_1[i])):
                    selected_1[i][j]=int(selected_1[i][j])
            for i in range(len(selected_2)):
                for j in range(len(selected_2[i])):
                    for k in range(len(selected_2[i][j])):
                        selected_2[i][j][k]=int(selected_2[i][j][k])
            print(selected_1)
            print(selected_2)
            #input()

            for num in range(cur_batch_size):
                data={'intent':all_intent[selected[num]],'layer1':selected_1[num],'layer2':selected_2[num],'key':test_batch.my_id[num]}
                generated_DAs.append(data)

            
            
        #candidate=candidates[0]
        #print(candidate)

        

        cur_batch_size=intent.size()[0]
        for num in range(cur_batch_size):
            
            # for mov in range(5):
            #     score.append(torch.sum(step[num,:]*candidate[idx]).item())
            #     idx+=1
            #if e2e==True:
            score=step[num,:]
            score_ex=rec[num,:]
            #print(torch.max(score_ex,0).indices)
            #print(golden[num])
            #print(score)

            wrong_count=0
            wrong_count_ex=0
            for item in score:
                if item>=score[golden[num]]:
                    wrong_count+=1
            
            for item in score_ex:
                if item>=score_ex[golden[num]]:
                    wrong_count_ex+=1
            
            tot_turns+=1
            if wrong_count<2:
                turn_1+=1
            if wrong_count<4:
                turn_3+=1

            if wrong_count_ex<2:
                correct.append(1)
                turn_1_ex+=1
            else:
                correct.append(0)
            if wrong_count_ex<4:
                turn_3_ex+=1

            if test_batch.last_turn[num]==1:
                tot_dialog+=1
                if wrong_count<2:
                    chat_1+=1
                if wrong_count<4:
                    chat_3+=1
                
                if wrong_count_ex<2:
                    chat_1_ex+=1
                if wrong_count_ex<4:
                    chat_3_ex+=1
                

                for i,item in enumerate(correct):
                    index=int((i/(turn+1e-5))*splits)
                    tot_split[index]+=1
                    if item==1:
                        accuracy_split[index]+=1
                turn=0
                correct=[]
            else:
                turn+=1
            #idx+=1#skip none node


            # selected=torch.softmax(intent,dim=-1).max(dim=-1).indices
            # da_distrib[selected[num]]=da_distrib[selected[num]]+1
            # if selected[num]==intent_label[num]:
            #     intent_accuracy+=1
        #print(tot_split)
        #print(accuracy_split)
        ##print(score)



        if j==eval_batch:
            break

        j+=1
    
    print("score_sample:",score)
    print("ex_score_sample:",score_ex)
    
    turn_1=turn_1/tot_turns
    turn_3=turn_3/tot_turns
    chat_1=chat_1/tot_dialog
    chat_3=chat_3/tot_dialog

    turn_1_ex=turn_1_ex/tot_turns
    turn_3_ex=turn_3_ex/tot_turns
    chat_1_ex=chat_1_ex/tot_dialog
    chat_3_ex=chat_3_ex/tot_dialog

    for i,item in enumerate(accuracy_split):
        accuracy_split[i]=item/tot_split[i]

    intent_accuracy=intent_accuracy/tot_turns


    #print("accuracy_split:",accuracy_split)
    #print('da_distribution:',da_distrib)
    print("intent accuracy:",intent_accuracy)
    # print("turn_1",turn_1)
    # print("turn_3",turn_3)
    # print("chat_1",chat_1)
    # print("chat_3",chat_3)
    print("turn_1",turn_1_ex)
    print("turn_3",turn_3_ex)
    print("chat_1",chat_1_ex)
    print("chat_3",chat_3_ex)
    
    if gen_DA==True:
        f=open(args['DA_save_path'],'w')
        json.dump(generated_DAs,f)
    
    if gen_utter==True:
        from generator import Generator

        gener=Generator(args['gen_conf'])
        with open(args['DA_save_path'],'r') as f:
            generated_DAs=json.load(f)
            
        generate_all={}

        dataset=GoRecDial(args['data_path'],flag="test")
        for i,item in tqdm(enumerate(generated_DAs)):
            DA=da_tree_serial(item,args['id2name'])
            if len(dataset[i].dialog_history)!=0:
                context=dataset[i].dialog_history[-1]
            else:
                context="hello"
            
            context=utter_lexical_gorecdial(context,args['mid2name'])
            gpt_in=context+" @ "+DA+" &"

            print(gpt_in.lower())
            generated=gener.generate(gpt_in.lower())
            print(utter_lexical_gorecdial(dataset[i].oracle_response,args['mid2name']))
            cur_turn={"generated":generated,"label":utter_lexical_gorecdial(dataset[i].oracle_response,args['mid2name'])}
            generate_all[item['key']]=cur_turn
            if (i+1) % 50 ==0:
                print("saving results...")
                with open(args['utter_save_path'],'w') as f:
                    json.dump(generate_all,f)
        print("saving results...")
        with open(args['utter_save_path'],'w') as f:
            json.dump(generate_all,f)
        
        lines = [item['generated'].strip() for item in generate_all.values()]
        bleu_array = []
        f1_array = []
        
        k=0
        for item in generate_all.values():
            k+=1
            ground, generate = [item['label']], item['generated']
            bleu_array.append(bleu(generate, ground))
            f1_array.append(f1_score(generate, ground))
        print("BLEU:",np.mean(bleu_array))
        print("F1:",np.mean(f1_array))


        tokenized = [line.split() for line in lines]
        for n in range(1, 6):
            cnt, percent = distinct_n_grams(tokenized, n)
            print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')
        
        
    return intent_accuracy,turn_1,turn_3,chat_1,chat_3,turn_1_ex,turn_3_ex,chat_1_ex,chat_3_ex

