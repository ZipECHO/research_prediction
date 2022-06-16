import pickle
import torch
import random
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def load_file(file_path):
    """load file with 'rb' type

    Args:
        file_path (str): file path

    Returns:
        file obj: file load result
    """
    with open(file_path, 'rb') as file:
        case_paths = pickle.load(file)
        file.close()
    return case_paths

def save_pickle(obj, filepath):
    with open(filepath,'wb') as file:
        pickle.dump(obj,file,protocol=2)
        file.close()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_papers_dict(paper_path='data/arxiv-metadata-oai-snapshot.json'):
    with open(paper_path,'r') as file:
        lines = file.readlines()
        file.close()
    papers = []
    for line in tqdm(lines):
        dic = json.loads(line)
        papers.append(dic)
    papers = sorted(papers,key=lambda e:e.__getitem__('update_date'))
    return papers


def draw_count(count_list):
    plt.hist(count_list,bins=100)
    plt.show()