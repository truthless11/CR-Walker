import sys
import torch
import json
import torch.nn as nn
import random
import os.path as osp
import os
from termcolor import colored
from tqdm import tqdm
import spacy
from transformers import BertModel,BertTokenizer
from torch_geometric.data import DataLoader
from fuzzywuzzy import fuzz


model = "en_core_web_sm"
# model = "en"
print('spacy loading', model)
nlp = spacy.load(model)

dataset="redial"

root=osp.dirname(osp.dirname(osp.abspath(__file__)))
path=osp.join(root,"data","redial")
global_graph_path=osp.join(path,"raw",'redial_kg.json')
match_path=osp.join(path,"raw",'match_results')
save_path=osp.join(path,"raw",'match_results_new')
f=open(global_graph_path)
graph=json.load(f)
nodes=graph["nodes"]
relations=graph["relations"]

print(len(nodes))
# print(len(relations))

person_dict={}
subject_dict={}
genre_dict={}
general_dict={}
movie_dict={}
time_dict={}

genre_match={"Action":["action"],"Adventure":["adventure"],"Animation":['anime','animation','animated','cartoon'],"Children":['child','kids','children'],"Comedy":['comedy','funny','humor','comedies'],"Crime":['crime'],"Documentary":['documentary','documentaries'],"Drama":['drama'],"Fantasy":['fantasy','fantasies'],"Film-Noir":['film noir','neo noir'],"Horror":['horror','scary'],"Musical":['musical','musician'],"Mystery":['mystery','mysteries'],"Romance":['romance','romantic'],"Sci-Fi":['sci fi','sci-fi','science fiction','fiction'],"Thriller":['thriller','thrilling'],"War":['war'],"Western":['western']}

general_match={"Movie":[],"Actor":["actor","actress"],"Director":["director"],"Genre":["genre","type of movie","kind of movie"],"Time":["time","era","decade","older","newer"],"Attr":[],"Subject":["subject"],"None":[]}
time_match={"1900":["190"],"1910":["191"],"1920":["192","20s","20's"],"1930":["193","30s","30's"],"1940":["194","40s","40's"],"1950":["195","50s","50's"],"1960":["196","60s","60's"],"1970":["197","70s","70's"],"1980":["198","80s","80's"],"1990":["199","90s","90's"],"2000":['200'],"2010":['201'],"2020":['202']}

for i,node in enumerate(nodes):
    if node['type']=="Movie":
        idx=int(node['MID'])
        movie_dict[idx]=i
    if node['type']=="Person":
        name=node['name'].replace("_"," ").lower()
        person_dict[name]=i
    if node['type']=="Subject":
        name=node['name'].replace("_"," ").lower()
        subject_dict[name]=i
    if node['type']=="Genre":
        name=node['name']
        genre_dict[name]=i
    if node['type']=="Attr":
        name=node['name']
        general_dict[name]=i
    if node['type']=="Time":
        name=node['name'].lower().replace("s","")
        time_dict[name]=i


def match_movie(utterance):
    def get_idx(tok):
        nums="0123456789"
        result=""
        for s in tok:
            if s in nums:
                result+=s
        try:
            return int(result)
        except:
            return ""
    
    results=[]
    refined_utterance=""

    if dataset=="redial":
        tokens=utterance.split()
        for token in tokens:
            if "@" in token:
                idx=get_idx(token)
                if idx!="" and idx in movie_dict.keys():
                    results.append(movie_dict[idx])
                refined_utterance+="MOV"+" "
            else:
                refined_utterance+=token+" "
    else:
        if "RECOMMEND" in utterance:
            tokens=utterance.split()
            for i,token in enumerate(tokens):
                if "MID" in token and tokens[i-1]=="RECOMMEND":
                    idx=get_idx(token)
                    if idx!="" and idx in movie_dict.keys():
                        results.append(movie_dict[idx])
    return results,refined_utterance



def fuzzy_match_person(name):
    for item in person_dict.keys():
        score=fuzz.partial_ratio(item.replace("_"," "),name.lower())
        if score>85:
            return item
    return None

def fuzzy_match_subject(name,threshold1=80,threshold2=90):
    if len(name)>8 or name=="Marvel" or name=="Netflix" or name=="Disney" or name=="Pixar":
        for item in subject_dict.keys():
            score=fuzz.ratio(item.replace("_"," "),name.lower())
            score_part=fuzz.partial_ratio(item.replace("_"," "),name.lower())
            if score>threshold1 or score_part>threshold2 :
                return item
    elif len(name)<5:
        if name=="Love":
            return None
        for item in subject_dict.keys():
            score=fuzz.ratio(item.replace("_"," "),name.lower())
            if score>95:
                return item
    else:
        for item in subject_dict.keys():
            score=fuzz.ratio(item.replace("_"," "),name.lower())
            if score>threshold1:
                return item

    return None

def fuzzy_match_general(utter):
    match_list=[]
    for key,value in general_dict.items():
        for item in general_match[key]:
            if item in utter.lower():
                match_list.append(value)
                break
    return match_list

def fuzzy_match_genre(utter):
    match_list=[]
    for key,value in genre_dict.items():
        for item in genre_match[key]:
            if item in utter.lower():
                match_list.append(value)
                break
    return match_list

def fuzzy_match_time(utter):
    match_list=[]
    for key,value in time_dict.items():
        for item in time_match[key]:
            if item in utter.lower():
                match_list.append(value)
                break
    return match_list



def fuzzy_match_mentioned(name,mentioned):
    #print(colored(name,"blue"))
    for item in mentioned:
        #print(nodes[item]['name'])
        score=fuzz.partial_ratio(name.lower(),nodes[item]['name'].replace("_"," ").lower())
        #print(score)
        if score>90:
            return item
    #input()
    return None


# ProRec Template Entity Linker:
#1.regular expression to match General + Genre
#2.NER from Spacy to locate person entities
#3.el by Fuzzywuzzy for full name in person set
#4.el by Fuzzywuzzy for partial name in the mentioned set

def match_nodes(utterance,mentioned):
    matched=[]

    movie,refined=match_movie(utterance)
    matched=matched+movie

    general=fuzzy_match_general(refined)
    matched=matched+general

    genre=fuzzy_match_genre(refined)
    matched=matched+genre

    time=fuzzy_match_time(refined)
    matched=matched+time


    with nlp.disable_pipes("tagger", "parser"):
        tgt = nlp(utterance)
        for ent in tgt.ents:
            if ent.label_=="PERSON" or ent.label_=="ORG":
                if ent.text.lower() in person_dict.keys():
                    matched.append(person_dict[ent.text.lower()])
                    continue
                if ent.text.lower() in subject_dict.keys():
                    matched.append(subject_dict[ent.text.lower()])
                    continue

                if " " in ent.text:
                    key=fuzzy_match_person(ent.text)
                    if key!=None:
                        matched.append(person_dict[key])
                        continue
                key=fuzzy_match_mentioned(ent.text,mentioned)
                if key!=None:
                    matched.append(key)
                    continue
                key=fuzzy_match_subject(ent.text)
                if key!=None:
                    matched.append(subject_dict[key])
    return matched




# stopwords=['@','Netflix',"Marvel","Xmen","Harry Potter","Comedy","Thor","Genre","Sci Fi","Loki","Sci-Fi","Cartoon","Alien","Witch","Sci-fi","Merry Christmas","Merry Xmas","Sci"]

#print(len(person_dict.keys()))


# train_size=10006
# test_size=1341

# tot=0
# matched=0
# cnt_old=0
# cnt=0

# flags=['train','test']

# for q,flag in enumerate(flags):
#     data_list = []
#     tot=0
#     if flag=='train':
#         size=train_size
#     else:
#         size=test_size
#     for dialog_num in tqdm(range(size)):
#         #print(dialog_num)
#         #print("-----------------------------------------------------------")
#         my_match_path=osp.join(match_path,flag,flag+str(dialog_num)+".json")
#         my_save_path=osp.join(save_path,flag,flag+str(dialog_num)+".json")
#         f=open(my_save_path)
#         match=json.load(f)

#         mentioned=[]
#         for idx,item in enumerate(match):
#             utter=item["sentence"]["text"]
#             #print(mentioned)
#             matched=match_nodes(utter,mentioned)
#             for n in matched:
#                 if n not in mentioned:
#                     mentioned.append(n)
#                 #if len(matched)!=len(item["matched_nodes"]):
#                 #print(colored(utter,"red"))
#             flag1=0
#             for n in matched:
#                 flag1=1
#                 for ent_m in item["matched_nodes"]:
#                     if ent_m['global']==n:
#                         flag1=0
#                 if flag1==1:
#                     match[idx]["matched_nodes"].append(nodes[n])
#         f=open(my_save_path,'w')
#         json.dump(match,f)



                # flag1=0
                # for ent_m in item["matched_nodes"]:
                #     flag1=1
                #     print(colored(ent_m["name"],"blue"),end=" ")
                # if flag1==1:
                #     print()

#         seeker_id=match[0]["initiatorWorkerId"]
#         mentioned=[]
#         for idx,item in enumerate(match):
#             for ent_m in item["matched_nodes"]:
#                 if ent_m['global'] not in mentioned:
#                     mentioned.append(ent_m['global'])

#             if "funny" in item["sentence"]["text"].lower() or "humor" in item["sentence"]["text"].lower() or "hilirous" in item["sentence"]["text"].lower():
#                 #print(colored(item["sentence"]["text"],"green"))
#                 flag1=0
#                 for ent_m in item["matched_nodes"]:
#                     if ent_m['name']=="Comedy":
#                         flag1=1
#                 if flag1==0:
#                     match[idx]["matched_nodes"].append(nodes[30438])
#                     #print(nodes[30438]['name'])


#             elif "scary" in item["sentence"]["text"].lower():
#                 #print(colored(item["sentence"]["text"],"red"))
#                 flag1=0
#                 for ent_m in item["matched_nodes"]:
#                     if ent_m['name']=="Horror":
#                         flag1=1
#                 if flag1==0:
#                     match[idx]["matched_nodes"].append(nodes[30444])
#                     #print(nodes[30444]['name'])
                    
#             elif "sci-fi" in item["sentence"]["text"].lower() or "sci fi" in item["sentence"]["text"].lower() or "fiction" in item["sentence"]["text"].lower():
#                 #print(colored(item["sentence"]["text"],"blue"))
#                 flag1=0
#                 for ent_m in item["matched_nodes"]:
#                     if ent_m['name']=="Sci-Fi":
#                         flag1=1
#                 if flag1==0:
#                     match[idx]["matched_nodes"].append(nodes[30448])
#                     #print(nodes[30448]['name'])
#             elif "kids" in item["sentence"]["text"].lower() or "child" in item["sentence"]["text"].lower():
#                 flag1=0
#                 #print(colored(item["sentence"]["text"],"green"))
#                 for ent_m in item["matched_nodes"]:
#                     #print(ent_m['name'])
#                     if ent_m['name']=="Children":
#                         flag1=1
#                 if flag1==0:
#                     match[idx]["matched_nodes"].append(nodes[30437])
#             elif "anime" in item["sentence"]["text"].lower() or "animated" in item["sentence"]["text"].lower() or "cartoon" in item["sentence"]["text"].lower():
#                 flag1=0
#                 #print(colored(item["sentence"]["text"],"red"))
#                 for ent_m in item["matched_nodes"]:
#                     #print(ent_m['name'])
#                     if ent_m['name']=="Animation":
#                         flag1=1
#                 if flag1==0:
#                     match[idx]["matched_nodes"].append(nodes[30436])
#             elif "romantic" in item["sentence"]["text"].lower():
#                 flag1=0
#                 #print(colored(item["sentence"]["text"],"blue"))
#                 for ent_m in item["matched_nodes"]:
#                     #print(ent_m['name'])
#                     if ent_m['name']=="Romance":
#                         flag1=1
#                 if flag1==0:
#                     match[idx]["matched_nodes"].append(nodes[30447])

#         #             #print(ent_m['name'])
#         # f=open(my_save_path,'w')
#         # json.dump(match,f)

            
            
            

#         #     #speaker_id=item["sentence"]["senderWorkerId"]
#         #     # if speaker_id==seeker_id:
#         #     #     print("Seeker:",end="")
                
#         #     # else:
#         #     #     print("Recommender:",end="")

#         #     # if "funny" in item["sentence"]["text"]:
#         #     #     print(item["sentence"]["text"])

#             with nlp.disable_pipes("tagger", "parser"):
#                 tgt = nlp(item["sentence"]["text"])
#             for ent in tgt.ents:
#                 matched_node=None
#                 a=0
#                 for k in stopwords:
#                     if k in ent.text:
#                         a=1
#                         break
#                 if a==1:
#                     continue

#                 if ent.label_=="PERSON":
#                     flag_m=0
#                     #print(colored(ent.text,"red"), ent.label_)
#                     for ent_m in item["matched_nodes"]:
#                         if ent_m['type']=="Person":
#                             score=fuzz.partial_ratio(ent_m['name'].replace("_"," "),ent.text)
#                             if score>90:
#                                 flag_m=1
#                                 break
#                     if ent.text.lower() in person_dict.keys() and flag_m==0:
#                         flag_m=1
#                         matched_node=nodes[person_dict[ent.text.lower()]]
#                         #print(colored(ent.text,"green"),matched_node)
#                     if flag_m==0:
#                         if " " in ent.text:
#                             key=fuzzy_match_person(ent.text)
#                             if key!=None:
#                                 matched+=1
#                                 matched_node=nodes[person_dict[key]]
#                         else:
#                             key=fuzzy_match_mentioned(ent.text,mentioned)
#                             if key!=None:
#                                 flag1=0
#                                 #print(item["sentence"]["text"])
#                                 for ent_m in item["matched_nodes"]:
#                                     if ent_m['global']==key:
#                                         flag1=1
#                                 #print(colored(ent.text,"green"),matched_node)
#                                 if flag1==0:
#                                     matched_node=nodes[key]
#                                 matched+=1
#                 else:
#                     print(ent.text,ent.label_)
#                 # else:
                #     print(ent.text)
                            # else:
                            #     key=fuzzy_match_subject(ent.text)
                            #     if key!=None:
                            #         print(item["sentence"]["text"])
                            #         print(colored(ent.text,"green"),nodes[subject_dict[key]])
                            #     else:
                            #         print(colored(ent.text,"red"))
                                
                


                                


                                #print(colored(ent.text,"green"),matched_node)
                            #else:
                                #print(colored(ent.text,"red"), ent.label_)
                    
                    # if flag_m==1:
                    #     #print(colored(ent.text,"red"), ent.label_)
                    #     matched+=1
                

                # if matched_node!=None:
                #     flag1=0
                #     #match[idx]["matched_nodes"].append(matched_node)
                #     #print(item['sentence']['text'])
                #     for ent_m in item["matched_nodes"]:
                #         if ent_m['global']==matched_node['global']:
                #             flag1=1
                #         #print(ent_m['name'])
                #     if flag1==0:
                #         match[idx]["matched_nodes"].append(matched_node)
                        #print(colored(matched_node['name'],"green"))
                    #for item in match[idx]["matched_nodes"]:
                        #print()
                   
                #     print()

                #     tot+=1
        # f=open(my_save_path,'w')
        # json.dump(match,f)
            #if tot!=0:
                #print(matched/tot)

                            #print(colored(ent_m['name'],"green")," ",score)
                        # if ent['type']=="Genre":
                        #     print(colored(ent['name'],"blue"))
                    #print(ent.text, ent.start_char, ent.end_char, ent.label_)
        #print("-----------------------------------------------------------")     