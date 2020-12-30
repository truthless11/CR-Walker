# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 00:06:23 2020

@author: truthless
"""
import re

def da_serial(item, id2name):
    da_seq = item['intent'] + ' ('
    first_path = True
    for path in item['path']:
        if not first_path:
            da_seq += ' ;'
        first_path = False
        first_node = True
        for num in path:
            if not first_node:
                da_seq += ' ~'
            first_node = False
            da_seq += ' ' + id2name[str(num)].replace('_', ' ')
    da_seq += ' )'
    return da_seq

def da_tree_serial(item, id2name):
    da_seq = item['intent']
    for first, second in zip(item['layer1'], item['layer2']):
        da_seq += ' ( ' + id2name[str(first)].replace('_', ' ')
        if second:
            for node in second:
                da_seq += ' ( ' + id2name[str(node)].replace('_', ' ') + ' )'
        da_seq += ' )'
    return da_seq

def utter_lexical_gorecdial(da_uttr, mid2name):
    pattern = re.compile(r'RECOMMEND MID(\d+)')
    while pattern.search(da_uttr):
        mid = pattern.search(da_uttr).group(1)
        movie_name = mid2name[mid]
        da_uttr = da_uttr.replace('MID'+mid, movie_name)
    return da_uttr

def utter_lexical_redial(da_uttr, mid2name):
    pattern = re.compile(r'@(\d+)')
    while pattern.search(da_uttr):
        mid = pattern.search(da_uttr).group(1)
        movie_name = mid2name[mid]
        da_uttr = da_uttr.replace('@'+mid, movie_name)
    return da_uttr

def utter_lexical_redial_dcr(da_uttr, mid2name):
    pattern = re.compile(r'<(.*?)>')
    while pattern.search(da_uttr):
        raw = pattern.search(da_uttr).group(1)
        if 'MID' in raw:
          mid=raw[3:]
          movie_name = mid2name[mid]
          da_uttr = da_uttr.replace('<'+raw+'>', movie_name)
        else:
          name=raw.replace('_',' ')
          da_uttr = da_uttr.replace('<'+raw+'>', name)
    return da_uttr

def utter_lexical_redial_kbrd(da_uttr, mid2name):
    pattern = re.compile(r'\"(.*?)\(\d+\)\"')
    while pattern.search(da_uttr):
        raw = pattern.search(da_uttr).group(1)
        print(raw)
        #print(pattern.search(da_uttr).group(0))
        # print(raw)
        # if 'MID' in raw:
        #   mid=raw[3:]
        #   movie_name = mid2name[mid]
        #   da_uttr = da_uttr.replace(pattern.search(da_uttr).group(0), movie_name)
        # else:
        #   name=raw.replace('_',' ')
        da_uttr = da_uttr.replace(pattern.search(da_uttr).group(0), pattern.search(da_uttr).group(1))
        #da_uttr = da_uttr.replace('@'+mid, movie_name)

    return da_uttr

if __name__ == '__main__':
    import json
    '''gorecdial'''
    with open('id2name_gorecdial.json', 'r') as f:
        id2name = json.load(f)
    turn = {
    "utterance": "Do you have a favorite comedy director or actor?",
    "intent": "question",
    "path": [
      [
        11748
      ],
      [
        11747
      ],
      [
        11744,
        11725
      ]
    ]
  }
    print(da_serial(turn, id2name))
    '''redial'''    
    with open('id2name_redial.json', 'r') as f:
        id2name = json.load(f)
    turn = {
    "label": "Yes @151656 is very funny and so is @94688",
    "context": [
      "Hi I am looking for a movie like @111776",
      "You should watch @151656",
      "Is that a great one? I have never seen it. I have seen @192131. I mean @134643"
    ],
    "intent": "recommend",
    "layer1": [
      2648,
      6473
    ],
    "layer2": [
      [
        30438,
        30458
      ],
      [
        30438,
        30458
      ]
    ]
  }
    print(da_tree_serial(turn, id2name))
