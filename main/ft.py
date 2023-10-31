
import os
import sys
import data
import time
import pickle
import random
import traceback
import itertools
import threading

import torch
import numpy as np
from torch import nn
from mlp import MLP
from functools import wraps
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
    log_loss, auc, precision_recall_curve
    
import warnings
warnings.simplefilter("ignore")

def train_step(model, batch, device, loss_fn, opt, history):
    model.train()    
    loss = loss_fn(model(batch[0].to(device), batch[1].to(device)), 
                   batch[2].to(device).unsqueeze(1))
    opt.zero_grad()
    loss.backward()
    opt.step()

    history['bce'].append(loss.cpu().detach().item())
    return history

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def test_epoch(model, dataloader, device, history):
    y_truths = np.array([])
    y_preds = np.array([])
    model.eval()

    for Xc, Xd, y in dataloader:
        Xc = Xc.to(device)
        Xd = Xd.to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, y.cpu().detach().numpy().ravel()])
            y_pred = model(Xc, Xd).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])
#    print(y_truths, y_preds)
    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds))
    history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
    history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['bce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))

    return history

def model_save_check(history, metric_name, tolerance_count=10):
    # Exit when model performance poor after over tolerance_count times
    save_flag = False
    stop_flag = False
    if 'best_index' not in history:
        history['best_index'] = 0
        history['bad_count'] = 0
        return True, False
    if metric_name.endswith('loss'):
        if history[metric_name][-1] < history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
            history['bad_count'] = 0
        else:
            history['bad_count'] += 1
    else:
        if history[metric_name][-1] > history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
            history['bad_count'] = 0
        else:
            history['bad_count'] += 1

    if history['bad_count'] > tolerance_count and history['best_index'] > 0:
        stop_flag = True
    if stop_flag:
        history['bad_count'] = 0

    return save_flag, stop_flag


class Model(nn.Module):
    def __init__(self, sampleEncoder, drugEncoder, classifier):
        super(Model, self).__init__()
        self.se = sampleEncoder
        self.de = drugEncoder
        self.cf = classifier

    def forward(self, xc, xd):
        return self.cf(torch.cat((self.se(xc), self.de(xd)), dim=1)).sigmoid()

def fine_tune_model(sample_encoder, 
                    ccl_train_dataloader, 
                    ccl_test_dataloader,
                    pat_dataloader, 
                    **kwargs):
    sample_encoder = MLP(input_dim=2009, hidden_dims=[512, 256, 128, 64, 32], 
                         output_dim=64, dop=kwargs['dop'])
    drug_encoder = MLP(input_dim=next(iter(ccl_train_dataloader))[1].shape[1], 
                       output_dim=64, hidden_dims=[128, 64], dop=kwargs['dop'])
    classifier = MLP(input_dim=128,
                     output_dim=1,
                     hidden_dims=[128, 64], dop=kwargs['dop'])
    model = Model(sample_encoder, drug_encoder, classifier).to(kwargs['device'])
    
    classification_loss = nn.BCEWithLogitsLoss()

    target_classification_train_history = defaultdict(list)
    target_classification_eval_train_history = defaultdict(list)
    target_classification_eval_val_history = defaultdict(list)
    target_classification_eval_test_history = defaultdict(list)

    lr = kwargs['ft_lr']
#    lr = 0.005
    y_label = kwargs['y_label']
    model_save_name = os.path.join('./tmp', f'finetuned_model_{y_label}_{kwargs["ptName"]}.pt')

    train_para_list = [drug_encoder.parameters(), 
                       classifier.parameters(), 
                       sample_encoder.parameters(), 
                       ]
    target_classification_optimizer = torch.optim.AdamW(chain(*train_para_list),
                                                        lr=lr)
    encoder_module_indices = [i for i in range(len(list(sample_encoder.modules())))
                          if str(list(sample_encoder.modules())[i]).startswith('Linear')]

    for epoch in range(kwargs['train_num_epochs']):
        for step, batch in enumerate(ccl_train_dataloader):
            train_step(
                    model=model,
                    batch=batch,
                    loss_fn=classification_loss,
                    device=kwargs['device'],
                    opt=target_classification_optimizer,
                    history=target_classification_train_history)
        test_epoch(model=model,
                    dataloader=ccl_train_dataloader,
                    device=kwargs['device'],
                    history=target_classification_eval_train_history)
        test_epoch(
                model=model,
                  dataloader=ccl_test_dataloader,
                  device=kwargs['device'],
                  history=target_classification_eval_val_history)
        test_epoch(
                model=model,
                dataloader=pat_dataloader,
                device=kwargs['device'],
                history=target_classification_eval_test_history
                )
        save_flag, stop_flag = model_save_check(history=target_classification_eval_val_history,
                                                metric_name='auroc',
                                                tolerance_count=5,)
        if save_flag:
            torch.save(model.state_dict(), model_save_name)
        if stop_flag:
            if not encoder_module_indices:
                break
            ind = encoder_module_indices.pop()
            print(f'Unfreezing {epoch}')
            model.load_state_dict(torch.load(model_save_name))
            
            train_para_list.append(list(sample_encoder.modules())[ind].parameters())
            lr *= kwargs['decay_coefficient']
            target_classification_optimizer = torch.optim.AdamW(chain(*train_para_list), lr=lr)
#        if not epoch % 20:
        print(epoch)
        print(target_classification_train_history['bce'][-1], 
              target_classification_eval_val_history['best_index'])
        print('\t'.join(map(lambda x:f'{x:.6f}', (target_classification_eval_val_history[key][-1] for key in ['acc', 'auroc', 'aps', 'f1', 'bce', 'auprc']))))
        print('\t'.join(map(lambda x:f'{x:.6f}', (target_classification_eval_test_history[key][-1] for key in ['acc', 'auroc', 'aps', 'f1', 'bce', 'auprc']))))
                
    model.load_state_dict(torch.load(model_save_name))

    return model, (target_classification_train_history, target_classification_eval_train_history,
                   target_classification_eval_val_history, target_classification_eval_test_history)#, prediction_df

class NewThread(object):
    def __init__(self, max_thread=20):
        self.max_thread = max_thread

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            while True:
                func_thread_active_count = len([i for i in threading.enumerate() if i.name == func.__name__])
                if func_thread_active_count <= self.max_thread:
                    thread = threading.Thread(target=func, args=args, kwargs=kwargs, name=func.__name__)
                    thread.start()
                    break
                else:
                    time.sleep(1)
        return wrapper

#@NewThread(15)
def main(paraDict):
#    filePath = './ft'
#    goodPtModel = set()
#    for fileName in os.listdir(filePath):
#        with open(os.path.join(filePath, fileName), 'br') as file:
#                history = pickle.load(file)
#        if any(x>0.55 for x in history[2]['auroc']):
#            goodPtModel.add(fileName)
#    goodPtModel = {'model_' + x.split('_model_')[1] for x in goodPtModel}
    baseDict = {
               "dop":0.1, 
               "seed": 0, 
               "device": 0, 
               "es_flag": False, 
               "metric": "auroc", 
               "norm_flag": True, 
               "latent_dim": 128, 
               "y_label": y_label, 
               "retrain_flag": True, 
               "train_num_epochs":200, 
               "decay_coefficient": 0.5, 
               "model_save_folder": './', 
               "encoder_hidden_dims": [512, 256], 
               "input_dim": 2009, #next(iter(Xs_train))[0].shape[-1],
               }
    baseDict.update(paraDict)
    
    tag = "_".join(map(str, paraDict.values()))
    print('\n', tag)
    ds = data.FedTherapyDataSet(seed=0, 
                                batchSize=64, 
                                num_workers=1, 
                                shuffle=True)
    
    with open('./ae0/ae_0.0001_0.0_False_(5, 0).pkl', 'br') as file:
        encoder, *history = pickle.load(file)
    ccl_train_dataloader, ccl_test_dataloader, pat_dataloader = ds.load_labeled_data(0.8, y_label)
    model, ft_historys = fine_tune_model(
            encoder, 
            ccl_train_dataloader, 
            ccl_test_dataloader,
            pat_dataloader, 
            ptName = 'ae_'+tag, 
            **baseDict)
    with open(f'./ft/ftmodel_{y_label}_{tag}_ae', 'bw') as file:
        pickle.dump(ft_historys, file)
    
if __name__ == '__main__':
    y_label = 'auc'
    params_grid = {
#        "dop": [0., 0.1], 
#        "ft_lr": [0.1, 1e-2, 1e-3, 1e-4]
        "dop": [0], 
        "ft_lr": [0.1, 0.001]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
#    jump = 1
    for param_dict in update_params_dict_list:
#        if jump:
#            jump-=1
#            continue
        try:
            main(param_dict)
        except Exception as e:
            print(e)
            traceback.print_exc()