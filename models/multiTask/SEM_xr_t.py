# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

## 1017
import torch_explain as te
from torch_explain import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_explain.nn.concepts import ConceptReasoningLayer

__all__ = ['SEM']


class SEM(nn.Module):
    def __init__(self, args):
        super(SEM, self).__init__()
        
        ###  experiment
        ###########################
        ###########################
        self.embedding_size = args.emb_size
        ###########################
        ###########################
        
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        #self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out +123*self.embedding_size, args.post_fusion_dim)
        self.post_fusion_layer_1 = nn.Linear(args.text_out +41*self.embedding_size, args.post_fusion_dim)
        self.post_fusion_layer_7 = nn.Linear(123*self.embedding_size, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)
        
        self.post_fusion_layer_4 = nn.Linear(1,10)
        self.post_fusion_dropout_4 = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_5 = nn.Linear(10,10)
        self.post_fusion_layer_6 = nn.Linear(10,1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)
        
        
        
        

        
        # text concept layer
        # 1017
        # concept encoder
        self.text_concept_encoder = torch.nn.Sequential(
            torch.nn.Linear(args.text_out, 10),
            torch.nn.LeakyReLU(),
            te.nn.ConceptEmbedding(10, 41, self.embedding_size),
        )

        
        
         # audio concept layer
        # 1017
        # concept encoder
        self.audio_concept_encoder = torch.nn.Sequential(
            torch.nn.Linear(args.audio_out, 10),
            torch.nn.LeakyReLU(),
            te.nn.ConceptEmbedding(10, 41, self.embedding_size),
        )


        # video concept layer
        # 1017
        # concept encoder
        self.video_concept_encoder = torch.nn.Sequential(
            torch.nn.Linear(args.video_out, 10),
            torch.nn.LeakyReLU(),
            te.nn.ConceptEmbedding(10, 41, self.embedding_size),
        )
        
        self.task_predictor  = ConceptReasoningLayer(self.embedding_size,1)
        
        
               
        

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        
        # concept layer for text
        text_concept = self.text_concept_encoder(text)
        
        # concept layer for audio
        audio_concept = self.audio_concept_encoder(audio)
        
        # concept layer for video
        video_concept = self.video_concept_encoder(video)
        
      
        text_concept = torch.tensor(text_concept[0]).unsqueeze(0)
        #text_concept = torch.tensor(text_concept, dtype=torch.float32).unsqueeze(0)
        #text_concept = torch.tensor(text_concept)
        audio_concept = torch.tensor(audio_concept[0]).unsqueeze(0)  
        video_concept = torch.tensor(video_concept[0]).unsqueeze(0) 
        # fusion
        #fusion_h = torch.cat([text, audio, video], dim=-1)
        
        #print(video_concept.size())
        #fusion_h = torch.cat([text_concept,audio_concept,video_concept],dim=2)
        #dim1, dim2, dim3,dim4 = fusion_h.shape
        #fusion_h = fusion_h.view(dim2,dim3,self.embedding_size)
        #fusion_h = fusion_h.view(fusion_h.size(0), -1)
        
        
        #fusion_h = self.post_fusion_dropout(fusion_h)
        #print(fusion_h.size())
        #fusion_h = F.relu(self.post_fusion_layer_7(fusion_h), inplace=False)
        
        
        
        
        
        
        #print(fusion_h.size())
        # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        #print(text_h.size())
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)
        
        ## text concept
        #text_h = self.post_text_dropout(text_concept_reasoning)
        #text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        
        ## audio
        #audio_h = self.post_audio_dropout(audio_concept_reasoning)
        #audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        
        ## vision
        #video_h = self.post_video_dropout(video_concept_reasoning)
        #video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)
        



        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)
        
        # text cocept output
        text_concept_emb, text_concept_pred = self.text_concept_encoder(text)
        #print(text_concept_emb.size())
        #print(text_concept_pred.size())
        
        # audio concept output
        audio_concept_emb, audio_concept_pred = self.audio_concept_encoder(audio)
        #print(audio_concept_emb.size())
        #print(audio_concept_pred.size())  
              
        # video concept output
        video_concept_emb, video_concept_pred = self.video_concept_encoder(video)
        
        
        
        #c_emb = torch.cat((text_concept_emb, audio_concept_emb, video_concept_emb), dim=1)
        #c_pred = torch.cat((text_concept_pred, audio_concept_pred, video_concept_pred), dim=1)
        c_emb = text_concept_emb
        c_pred = text_concept_pred
        #print(c_emb.size())
        #print(c_pred.size()) 
        a,b,c = c_emb.shape
        cc_all = c_emb.view(a,b*c)
        
        # classifier-fusion
        #fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = text
        x_f_1 = self.task_predictor(c_emb,c_pred)
        #print(c_pred.size())
        fusion_all = torch.cat((fusion_h,cc_all), dim=1)
        fusion_h = self.post_fusion_dropout(fusion_all)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        #x_f_1 = self.task_predictor(c_emb,c_pred)
        #fusion_all = torch.cat((x_f_1, fusion_h), dim=1)
        #print('test point')
        #print(x_f_1.size())
        #print(fusion_h.size())
        #print(tensor.size())
        #x_f = self.post_fusion_layer_4(x_f)
        #x_f = self.post_fusion_dropout_4(x_f)
        #x_f = F.relu(self.post_fusion_layer_5(x_f), inplace=False)
        #output_fusion = self.post_fusion_layer_6(x_f)
        
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)
        
        
        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)
        
        def matrix_sum(matrix):
            total = 0
            for row in matrix:
                for element in row:
                    total += element
            return total
            
        t_s = matrix_sum(text_concept_emb)
        a_s = matrix_sum(audio_concept_emb)
        v_s = matrix_sum(video_concept_emb)
        
        #print(fusion_h.size())
        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            't_emb':text_concept_emb,
            't':text_concept_pred,
            'a_emb':audio_concept_emb,
            'a':audio_concept_pred,
            'v_emb':video_concept_emb,
            'v':video_concept_pred,
            'c_emb':c_emb,
            'm':c_pred,
            't_s':t_s,
            'a_s':a_s,
            'v_s':v_s,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
