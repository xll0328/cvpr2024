import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import re
import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

from torch_explain.nn.concepts import ConceptReasoningLayer

logger = logging.getLogger('MSA')

class SEM():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.args.tasks_ed = "M"
        self.args.tasks_c = 'tav'
        self.args.loss1 = args.loss1
        self.args.loss2 = args.loss2
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        ####
        self.args.embedding = args.emb_size
        ####
        
        self.fusion_tensor = torch.zeros(args.train_samples, self.args.embedding*3, requires_grad=False).to(args.device)
        #fusion_tensor = self.fusion_tensor.unsqueeze(1).expand(-1, 41, -1)
        self.feature_map = {
            'fusion': self.fusion_tensor.unsqueeze(1).expand(-1, 41, -1),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
            }
        }
        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        # ttt = torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            't':torch.zeros((args.train_samples, 41), requires_grad=False).to(args.device),
            'a':torch.zeros((args.train_samples, 41), requires_grad=False).to(args.device),
            'v':torch.zeros((args.train_samples, 41), requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision',
            't': 't',
            'a': 'a',
            'v': 'v'
        }

    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)               

                ################### 1017
                labels_concept = batch_data['labels']['concept']
                # print("labels_concept")
                # print(labels_concept.size())
                
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m,labels_concept)


        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        def matrix_sum(matrix):
            total = 0
            for row in matrix:
                for element in row:
                    total += element
            return total
        # loop util earlystop
        sss_t = 0
        sss_a = 0
        sss_v =0
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': [] , 't': [], 'a': [], 'v': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': [] , 't': [], 'a': [], 'v': []}
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            #sss_t = 0
            #sss_a = 0
            #sss_v = 0
            ss_t = 0
            ss_a = 0
            ss_v = 0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths.cpu()), (vision, vision_lengths.cpu()))
                    #value = torch.sum(outputs['t_s']).item()
                    
                    #print(value)
                    score_t = torch.sum(outputs['t_s']).item()
                    score_a = torch.sum(outputs['a_s']).item()
                    score_v = torch.sum(outputs['v_s']).item()
                    #print(score_t)
                    sss = score_t+score_a+score_v
                    #print(score_t)
                    score_t = score_t/sss
                    score_a = score_a/sss
                    score_v = score_v/sss
                    
                    ss_t = ss_t + score_t
                    ss_a = ss_a + score_a
                    ss_v = ss_v + score_v
                    
                    #print(score_t)
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                        
                    # compute loss
                    loss_l = 0.0
                    for m in self.args.tasks_ed:
                        #print(m)
                        #print(outputs[m].size())
                        #print(self.label_map[self.name_map[m]][indexes].size())
                        loss_l += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes], \
                                                    indexes=indexes, mode=self.name_map[m])
                    torch.cuda.empty_cache()
                    loss_c = torch.nn.BCELoss()
                    #print(outputs['t'].size())
                    #print(self.label_map[self.name_map['t']][indexes].size())
                    loss_t = loss_c(outputs['t'], self.label_map[self.name_map['t']][indexes])
                    loss_a = loss_c(outputs['a'], self.label_map[self.name_map['a']][indexes])
                    loss_v = loss_c(outputs['v'], self.label_map[self.name_map['v']][indexes])
                    
                    loss = loss_l*self.args.loss1+ (loss_a)*self.args.loss2
                    #loss = loss_l*self.args.loss1+ (loss_t + loss_a + loss_v)*self.args.loss2
                    
                    # loss for concept text
                    #loss_c_t = 0.0
                    #y_pred['t'].append(outputs['t'].cpu())
                    #y_true['t'].append(self.label_map[self.name_map['t']][indexes].cpu())
                    #loss_c_t = self.weighted_loss(outputs['t'], \
                    #self.label_map[self.name_map['t']][indexes], indexes=indexes, mode=self.name_map['t'])
 
                    # loss for concept audio
                    #loss_c_a = 0.0
                    #y_pred['a'].append(outputs['a'].cpu())
                    #y_true['a'].append(self.label_map[self.name_map['a']][indexes].cpu())
                    #loss_c_a += self.weighted_loss(outputs['a'], \
                    #self.label_map[self.name_map['a']][indexes], indexes=indexes, mode=self.name_map['a'])
                    
                    # loss for concept video
                    #loss_c_v = 0.0
                    #y_pred['v'].append(outputs['v'].cpu())
                    #y_true['v'].append(self.label_map[self.name_map['v']][indexes].cpu())
                    #loss_v_v += self.weighted_loss(outputs['v'], \
                    #self.label_map[self.name_map['v']][indexes], indexes=indexes, mode=self.name_map['v'])
                    
                                       
                    # backward
                    loss.backward()
                    #train_loss += loss.item() + loss_c_t + loss_c_a + loss_c_v
                    train_loss += loss.item()
                    # update features
                    f_fusion = outputs['Feature_f'].detach()
                    f_text = outputs['Feature_t'].detach()
                    f_audio = outputs['Feature_a'].detach()
                    f_vision = outputs['Feature_v'].detach()
                    if epochs > 1:
                        self.update_labels(f_fusion, f_text, f_audio, f_vision, epochs, indexes, outputs)

                    self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    self.update_centers()
                    torch.cuda.empty_cache()
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                    torch.cuda.empty_cache()
                #print("ssss1")
                #print(ss_t)
                ssss = ss_t+ss_a+ss_v
                sss_t = ss_t/ssss
                sss_a = ss_v/ssss
                sss_v = ss_a/ssss
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            print("epoch is",end='')
            print(epochs)
            print(sss_t)
            print(sss_a)
            print(sss_v)
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                return
            torch.cuda.empty_cache()

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': [] , 't': [], 'a': [], 'v': [] , 'c_emb':[], 'c_pred' : []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': [] , 't': [], 'a': [], 'v': []}
        eval_loss = 0.0
        score_a = []
        score_t = []
        score_v = []
        def matrix_addition(matrix1, matrix2):
            result = []
            for i in range(len(matrix1)):
                row = []
                for j in range(len(matrix1[i])):
                    row.append(matrix1[i][j] + matrix2[i][j])
                result.append(row)
            return result
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    #labels_c = batch_data['labels']['concept'].to(self.args.device).view(-1)
                    #print(labels_c)
                    #print(labels_m)
                    labels_c = batch_data['labels']['concept']
                    # print(labels_c)
                    outputs = model(text, (audio, audio_lengths.cpu().to(torch.int64)), (vision, vision_lengths.cpu().to(torch.int64)))
                    
                    loss_c = torch.nn.BCELoss()
                    
                    # loss_t = loss_c(outputs['t'], labels_C)
                    
                    
                    #loss = self.weighted_loss(outputs['M'], labels_m) + loss_c(outputs['t'], labels_c)+loss_c(outputs['a'], labels_c)+loss_c(outputs['v'], labels_c)
                    loss = self.weighted_loss(outputs['M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
                    y_pred['t'].append(outputs['t'].cpu())
                    y_true['t'].append(labels_m.cpu())
                    y_pred['a'].append(outputs['a'].cpu())
                    y_true['a'].append(labels_m.cpu())
                    y_pred['v'].append(outputs['v'].cpu())
                    y_true['v'].append(labels_m.cpu()) 
                    y_pred['c_emb'].append(outputs['c_emb'].cpu())                           
                    y_pred['c_pred'].append(outputs['m'].cpu())
                    
                    
        eval_loss = eval_loss / len(dataloader)
        loss_pp = torch.nn.BCELoss()
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        pred_t,true_t = torch.cat(y_pred['t']), torch.cat(y_true['t'])
        pred_a,true_a = torch.cat(y_pred['a']), torch.cat(y_true['a'])
        pred_v,true_v = torch.cat(y_pred['v']), torch.cat(y_true['v'])
        c_emb,c_pred = torch.cat( y_pred['c_emb']),torch.cat(y_pred['c_pred'])
        
        
        
        
        #instance = ConceptReasoningLayer.ClassName()
        #print(dir(model))
        #CRL = model.Model.task_predictor(x= c_emb[0], c = c_pred[0])
        local_explanations = model.Model.task_predictor.explain(x= c_emb[0].to(self.args.device), c = c_pred[0].to(self.args.device), mode='local')
        global_explanations = model.Model.task_predictor.explain(x = c_emb[0].to(self.args.device), c = c_pred[0].to(self.args.device), mode='global')
        print(local_explanations)
        print(global_explanations)
        #print("a sample")
        #print(pred.size())
        #print(true.size())
        #print(pred_t.size())
        #print(true_t.size())
        print("samples")
        print(pred[0])
        print(true[0])
        print(pred_t[0])
        print(true_t[0])
        print(pred_a[0])
        print(true_a[0])
        print(pred_v[0])
        print(true_v[0])
        #print(pred[1])
        #print(true[1])
        #print(pred_t[1])
        #print(true_t[1])
        #print(pred_a[1])
        #print(true_a[1])
        #print(pred_v[1])
        #print(true_v[1])
        #print(
        eval_results = self.metrics(pred, true)
        #eval_results_t = loss_pp(pred_t, true_t)
        #eval_results_a = loss_pp(pred_a, true_a)
        #eval_results_v = loss_pp(pred_v, true_v)
        logger.info('M: >> ' + dict_to_str(eval_results))
        #logger.info('t: >> ' + dict_to_str(eval_results_t))
        #logger.info('a: >> ' + dict_to_str(eval_results_a))
        #logger.info('v: >> ' + dict_to_str(eval_results_v))
        eval_results['Loss'] = eval_loss

        return eval_results
    
    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        #print(y_pred.size())
        #print(y_true.size())
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    
    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        #print(self.feature_map['fusion'][indexes].size())
        # self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        # update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')
    
    def init_labels(self, indexes, m_labels, labels_concept):
        # print(indexes)
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels
        self.label_map['t'][indexes]=labels_concept.to(self.args.device)
        self.label_map['a'][indexes]=labels_concept.to(self.args.device)
        self.label_map['v'][indexes]=labels_concept.to(self.args.device)
    
    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8

        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1) 
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1) 
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            # d_s_pn = torch.norm(self.center_map[mode]['pos'] - self.center_map[mode]['neg'], dim=-1)
            # delta_s = (d_sn - d_sp) / (d_s_pn + MIN)
            #print("delta_s")
            #print(delta_s.size())
            #print("delta_f")
            #print(delta_f.size())
            #print('MIN')
            #print(MIN)
            #delta_f = torch.mean(delta_f, dim=1, keepdim=True)
            #delta_f = torch.squeeze(delta_f)
            
            #print("delta_s")
            #print(delta_s.size())
            #print("delta_f")
            #print(delta_f.size())
            
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                        0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
            # new_labels = torch.tanh(new_labels) * self.args.H

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        #d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1) 
        #d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1) 
        # d_f_pn = torch.norm(self.center_map['fusion']['pos'] - self.center_map['fusion']['neg'], dim=-1)
        # delta_f = (d_fn - d_fp) / (d_f_pn + MIN)
        #delta_f_1 = (d_fn - d_fp) / (d_fp + MIN)
        #delta_f = torch.mean(delta_f_1, dim=-1, keepdim=True)
        #print(delta_f.size())
        #delta_f = torch.squeeze(delta_f)
        #print("delta_f")
        #print(delta_f.size())
        #update_single_label(f_text, mode='text')
        #update_single_label(f_audio, mode='audio')
        #update_single_label(f_vision, mode='vision')