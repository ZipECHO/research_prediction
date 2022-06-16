import torch
from utils import save_pickle,load_file
from config import *
import numpy as np
from torch.utils.data import DataLoader,Dataset,TensorDataset
from model.sp_model import sp_model
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from utils import setup_seed

import argparse

import logging


writer = SummaryWriter('runs/')

from tqdm import tqdm

def get_exps(config):
    if config.pred_entity == 'people':
        exps = list(load_file('data/flt_auth_exp.pickle').key_list)
    if config.pred_entity == 'journal':
        exps = list(load_file('data/align/flt_journal_key_list.pickle').key_list)
    return exps

def win_sample(config,l):
    res = []
    if len(l)<=config.window_size:
        res = l
        for i in range(config.window_size-len(l)):
            res = [pad_key]+res
        return [res]
    for i in range(0,len(l)-config.window_size,config.win_step):
        res += [l[i:i+config.window_size]]
    return res


def get_train_pad_sequences(config):
    exps = get_exps(config)
    train_pad_seq = []
    for exp in exps:
        train_pad_seq += win_sample(config,exp[:-1])
        # test_exp += exp[-1]
    return np.array(train_pad_seq)

def get_all_pad_sequences(config):
    exps = get_exps(config)
    all_pad_seq = []
    for exp in exps:
        all_pad_seq += win_sample(config,exp)
        # test_exp += exp[-1]
    return np.array(all_pad_seq)


def win_sample_last(config, exp):
    if len(exp) >= config.window_size:
        return [exp[-config.window_size:]]
    res = exp
    for _ in range(config.window_size - len(res)):
        res  = [pad_key]+res
    return [res]



def get_test_pad_exp(config):
    exps = get_exps(config)
    test_pad_seq = []
    for exp in exps:
        test_pad_seq += win_sample_last(config,exp)
    return np.array(test_pad_seq)

def split_get_dataloader(total_pad_seq,input_size,k_emb_size):
    k2emb = load_file('data/k2emb.pickle')
    k2idx = load_file('data/k2idx.pickle')
    train_exp = total_pad_seq[:,:input_size]
    y_exp = total_pad_seq[:,-1]
    train_emb = np.zeros([len(train_exp),input_size*k_emb_size])
    i = 0
    for train_exp_key in train_exp:
        emb = np.array([])
        for key in train_exp_key:
            if key == 'unk':
                emb = np.concatenate([emb,np.zeros(k_emb_size)])
            else:
                emb = np.concatenate([emb, k2emb[key]])
        train_emb[i] = emb
        i+=1
    y = np.zeros([len(y_exp),160])

    i=0
    for lab in y_exp:
        y[i,k2idx[lab]] = k2idx[lab]
        i+=1
    torch_dataset = TensorDataset(torch.Tensor(train_emb),torch.Tensor(y))
    dataloader = DataLoader(
        dataset=torch_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 1,
        drop_last=True,
    )
    return dataloader

def get_acc(y_hat,lab):
    t = 0
    for pred,l in zip(y_hat,lab):
        if pred==l:
            t += 1
    return t/len(y_hat)

def train(config):
    print(config)
    writer = SummaryWriter(
        f'runs/grid_search/pred_entity_{config.pred_entity}/window_size_{config.window_size}/win_step_{config.win_step}/num_epochs_{config.num_epochs}/')

    train_pad_seq = get_train_pad_sequences(config)
    input_size = config.window_size-1
    drop_rate = config.drop_rate
    train_dataloader = split_get_dataloader(train_pad_seq, input_size, k_emb_size)
    device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')
    model = sp_model(input_size,k_emb_size,drop_rate).to(device)
    opt =  torch.optim.Adam(model.parameters(),lr = config.lr,weight_decay=0.01)

    # loss = torch.nn.BCELoss()
    loss = torch.nn.CrossEntropyLoss()
    epoches = tqdm(range(config.num_epochs))
    # train_dataloader.to(device)
    it = 0
    model.train()
    for e in epoches:
        for batch_x,y in train_dataloader:
            opt.zero_grad()
            batch_x = batch_x.to(device)
            y = y.to(device)
            pred = model(batch_x)
            l = loss(pred,y)
            l.backward()
            opt.step()
            it += 1
            writer.add_scalar('loss',l.cpu().item(),it)
            writer.add_scalar('acc',get_acc(np.array(torch.argmax(pred,dim=1).cpu()),np.array(torch.argmax(y,dim=1).cpu())),it)
        epoches.set_postfix(Loss = l.cpu().item(),Acc=get_acc(np.array(torch.argmax(pred,dim=1).cpu()),np.array(torch.argmax(y,dim=1).cpu())))
        test(model,config,writer,step=e)
        torch.save(model.state_dict(), '{}_model.pt'.format(config.pred_entity))

    writer.close()

    return model


def test(model,config,writer,step = -1):
    input_size = config.window_size-1 #根据前面的input——size个输入预测下一个
    drop_rate = config.drop_rate
    test_pad_seq = get_test_pad_exp(config)
    test_dataloader = split_get_dataloader(test_pad_seq, input_size,k_emb_size)
    device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')

    y_hat = np.array([])
    lab = np.array([])
    model.eval()
    model.to(device)
    with torch.no_grad():
        pred_vecs = np.array([])
        for batch_x, y in test_dataloader:
            batch_x = batch_x.to(device)
            pred_vec = model(batch_x)
            pred_vecs = np.append(pred_vecs,np.array(pred_vec.cpu()))
            y_hat = np.concatenate([y_hat, np.array(torch.argmax(pred_vec, dim=1).cpu())])
            lab = np.concatenate([lab, np.array(torch.argmax(y, dim=1).cpu())])
        pred_vecs = pred_vecs.reshape([-1,160])
    if config.run_model == 'grid_search':

        writer.add_scalar('valid results',get_acc(y_hat,lab),step)
    else:
        # writer.add_text('valid results',f'acc:{metrics.top_k_accuracy_score(lab)},hit@3:{},hit@5:{},hit@10:{}')
        print('acc:{}'.format(metrics.accuracy_score(lab,y_hat)))
        print('hit@3:{}'.format(metrics.top_k_accuracy_score(lab,pred_vecs,k=3,labels = np.arange(160))))
        print('hit@5:{}'.format(metrics.top_k_accuracy_score(lab,pred_vecs,k=5,labels = np.arange(160))))
        print('hit@10:{}'.format(metrics.top_k_accuracy_score(lab,pred_vecs,k=10,labels = np.arange(160))))

    # writer.close()
    save_pickle(obj=(y_hat,lab),filepath='res/config.window_size_{}_num_epo_{}_test_res_pred.pickle'.format(config.window_size,config.num_epochs))

    print('test acc:{} '.format(get_acc(y_hat,lab)))

    return get_acc(y_hat,lab)

def get_entity_exp_dict(config):
    if config.pred_entity == 'people':
        data = load_file('data/flt_auth_exp.pickle')
        return data.set_index('auth').key_list.to_dict()
    if config.pred_entity == 'journal':
        data = load_file('data/align/flt_journal_key_list.pickle')
        return data.set_index('journal').key_list.to_dict()

def get_pad_seq_by_entity(config):
    # exps = get_exps(config)
    # index = get_index(config)
    entity_exp_dict = get_entity_exp_dict(config)
    all_ent_pad_seq_dict = {}
    for ent,exp in entity_exp_dict.items():
        # try:
        all_ent_pad_seq_dict[ent]=win_sample(config, exp)
        # test_exp += exp[-1]
    return all_ent_pad_seq_dict


def get_all_entity_tensor(config,all_pad_seq_dict):
    k2emb = load_file('data/k2emb.pickle')
    k2idx = load_file('data/k2idx.pickle')

    input_exp_entity = {}
    lab = {}

    for key,val in all_pad_seq_dict.items():
        input_exp_entity[key] = []
        lab[key] = []
        for exp in val:
            input_exp_entity[key].append(exp[:-1])
            lab[key].append(int(k2idx[exp[-1]]))

    input_emb_entity = {}
    # lab_emb = {}
    for key,val in input_exp_entity.items():
        input_emb_entity[key] = []
        pad_list = np.zeros(k_emb_size).tolist()
        for win_exps in val:
            temp = []
            for exp in win_exps:

                if exp == 'unk':
                    temp.append(pad_list)
                else:
                    temp.append(k2emb[exp])
            input_emb_entity[key].append(temp)

    return input_emb_entity, lab


def final_test_find_case(model,config):
    input_size = config.window_size - 1  # 根据前面的input——size个输入预测下一个
    drop_rate = config.drop_rate
    all_pad_seq_dict = get_pad_seq_by_entity(config)
    # test_dataloader = split_get_dataloader(all_pad_seq_dict, input_size, k_emb_size)
    all_tensor_dict,lab = get_all_entity_tensor(config,all_pad_seq_dict)
    device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')

    model.to(device)
    res_entity = {}
    count = 0
    total_c = 0
    for entity,win_lists in tqdm(all_tensor_dict.items()):



        # entity_pred = np.array([])

        input = torch.tensor(np.array(win_lists)).to(device)
        input = input.reshape([-1,input_size*k_emb_size]).to(torch.float32)


        try:
            entity_pred = model(input)
            total_c += entity_pred.shape[0]
        except:
            print(input)


        # for i,exp_emb in enumerate(win_lists):
        #     exp_emb = torch.tensor(np.array(exp_emb)).to(device)
        #     exp_emb = exp_emb.reshape([-1])
        #     pred = model(exp_emb)
        #     pred = pred.cpu().detach().numpy()
        #     entity_pred = np.append(entity_pred,pred)
        entity_pred = entity_pred.reshape([-1,160])
        # try:
        def hit(y_true,y_score,k=2,labels=np.arange(160)):
            h=0
            top_k_indices = y_score.topk(k,dim=1)[1]
            for i in range(len(y_true)):
                if y_true[i] in top_k_indices[i]:
                    h+=1
            return float(h)/len(y_true)

        acc = hit(lab[entity],entity_pred,k=1,labels=np.arange(160))
        top_2 = hit(lab[entity],entity_pred,k=2,labels = np.arange(160))
        top_5 = hit(lab[entity],entity_pred,k=5,labels = np.arange(160))
        # print(len(lab[entity]))
        # print(entity_pred.shape)
        res_entity[entity] = (acc,top_2,top_5)
            # break
        # except:
            # print(len(lab[entity]))
            # print(entity_pred.shape)
            # count += 1
            # break
    # print(count/total_c)
    save_pickle(obj=res_entity,filepath='res/{}_exp_result.pickle'.format(config.pred_entity))
    print('test on {} done! All results saved in res/{}_exp_result.pickle!'.format(config.pred_entity,config.pred_entity))
    return res_entity


def grid_search(config):
    for ws in range(5, 41, 5):
        for ss in range(2, 10, 2):
            if ss >= ws:
                continue
            # for ne in range(5,0,)
            config.window_size = ws
            config.win_step = ss
            model = train(config)
    return



def main(config):
    setup_seed(20)
    print('run with :',end=' ')
    print(config)
    logging.info(config)
    if config.pred_entity == 'people':
        if config.search_best_ws:
            config.window_size = 5
            for _ in range(5):
                print('train with window size of:{}'.format(config.window_size))
                model = train(config)
                test(model,config)
                config.window_size+=1
        if config.run_model == 'train':
            model = train(config)
            # test(model,config.window_size,writer)
    if config.run_model == 'case_study':
        # model = train(config)
        # torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')
        input_size = config.window_size - 1
        drop_rate = config.drop_rate
        model = sp_model(input_size, k_emb_size, drop_rate).to(device)
        model.load_state_dict(torch.load('{}_model.pt'.format(config.pred_entity)))
        final_test_find_case(model, config)


    if config.run_model == 'grid_search':
        grid_search(config)

    if config.run_model == 'train':
        if config.pred_entity == 'journal':
            model = train(config)

    return




if __name__ == '__main__':

    # logging.basicConfig(filename='runs/logs.txt', filemode='a', level='DEBUG')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--window_size',type=int,default=20)
    # parser.add_argument('--pred_entity',type=str,default = 'journal')
    # parser.add_argument('--win_step',type=int,default=4)
    # parser.add_argument('--lr',type=float,default= 1e-3)
    # parser.add_argument('--search_best_ws',type=bool,default=False)
    # parser.add_argument('--drop_rate',type=float,default=1e-2)
    # parser.add_argument('--num_epochs',type = int,default=40)
    # parser.add_argument('--run_model',type=str,default='case_study')
    # config = parser.parse_args(args=[])
    # writer_filename_list = [item for item in config.__dict__.items()]

    # main(config)
    print('test')
