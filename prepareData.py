#!/usr/bin/python3
import argparse

import sys, os
# import nltk
import time
import stanza
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, help = 'Folder containing documents to be summarized')
parser.add_argument('--prep_path',  type = str, help = 'Path to store the prepared data in json format')
parser.add_argument('--lang', type = str, help="choose ta for tamil and en for english")

args = parser.parse_args()

BASEPATH = args.data_path
writepath = args.prep_path
lang = args.lang

stanza.download("en")
stanza.download("ta")

nlp_en = stanza.Pipeline("en", processors="tokenize,pos")
nlp_ta = stanza.Pipeline("ta", processors="tokenize,pos")


separator = "\t"

FILES = []
FILES2 = os.listdir(BASEPATH)
for f in FILES2:
        FILES.append(f)
DATA_FILES = {}
for F in FILES:
    start = time.time()
    ifname = os.path.join(BASEPATH,F)
    
    #print(F)
    fp = open(ifname,'r')
    dic = {}
    lines = fp.read().split("\n&\n")
    for l in lines:
        # print(f"Line {l}")
        try:
            wl = l.split(separator)
            # print("debug" , len(wl))
            CL = wl[1].strip(' \t\n\r')
            TEXT = wl[0].strip(' \t\n\r')
            
            # TEXT = TEXT.replace("sino noindex makedatabase footer start url", "")
            if TEXT:
                words = TEXT.split()
                if words and sum(1 for i in words if len(i) == 1)/len(words) >= 0.6:
                    continue
                if dic.__contains__(CL)==True:
                    temp = dic[CL]
                    temp.append(TEXT)
                    dic[CL] = temp
                else:
                    dic[CL] = [TEXT]
        except Exception as e:
            print(e)
    
    f_d = {}
    nlp = nlp_en if lang == "en" else nlp_ta

    for cl, sentences in dic.items():
        temp = []
        for s in sentences:
            doc = nlp(s)
            tokens = [word.text for sent in doc.sentences for word in sent.words]
            pos_tags = [(word.text, word.xpos) for sent in doc.sentences for word in sent.words]
            temp.append((s, tokens, pos_tags))
        f_d[cl] = temp



    DATA_FILES[F.split('.txt')[0].strip(' \t\n\r')] = f_d
    print('Complete {}'.format(F))
    print(f'Time : {time.time() - start}')

with open(os.path.join(writepath,'prepared_data.json'),'w') as legal_f:
    json.dump(DATA_FILES,legal_f,indent=4)
