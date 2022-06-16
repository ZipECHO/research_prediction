from utils import *
import json
import re
from fuzzywuzzy import fuzz

def get_j_list(j_l_path = 'journal_lists.txt'):
    with open(j_l_path,'r') as file:
        lines = file.readlines()
        j_list = []
        for line in lines:
            line = line[:-1].split('\t')
            j_list.append(line)
        file.close()
    return j_list

def j_name_align(s,j_list):
    m = 0
    j_name = 0
    for j in j_list:
        for name in j:
            sim = fuzz.ratio(s,name)
            if sim>m:
                m = sim
                j_name = j[0]
    return j_name

def align(papers,j_list):
    key_dict_conf = {}
    no_j_c = 0
    count = 0
    papers_align = papers.copy()
    for paper in tqdm(papers):
        journal_conf = paper['journal-ref']
        if journal_conf:
            j_s = paper['journal-ref']
            spls = re.findall(r'[0-9]|,|\n|\(|-|\\', j_s)
            if spls:
                journal_conf = j_s.split(spls[0])[0]
                journal_conf = j_name_align(journal_conf, j_list)
            if journal_conf not in key_dict_conf.keys():
                key_dict_conf[journal_conf] = []
            papers_align[count]['journal-ref'] = journal_conf
            key_dict_conf[journal_conf] += paper['categories'].split(' ')
        else:
            no_j_c += 1
        count+=1
        if count%10000==0:
            save_pickle(obj = papers_align,filepath='papers_align.pickle')
            print("Finished {} paper align file saved as papers_align.pickle".format(count))

if __name__ =="__main__":
    papers = get_papers_dict()
    j_list = get_j_list()
    align(papers,j_list)


