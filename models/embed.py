import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        Embed = nn.Embedding
        self.route_id_embed = Embed(2602, d_model)
        self.station_id_embed = Embed(37689, d_model)
        self.direction_embed = Embed(2, d_model)
        self.hour_embed = Embed(24, d_model)
    
    def forward(self, x):
        x = x.long()
        route_id_x = self.route_id_embed(x[:,:,0])
        station_id_x = self.station_id_embed(x[:,:,1])
        direc_id_x = self.direction_embed(x[:,:,2])
        hour_id_x = self.hour_embed(x[:,:,3])
        
        return route_id_x + station_id_x + direc_id_x + hour_id_x
        # return route_id_x + station_id_x + direc_id_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()

        self.embed = nn.Linear(2, d_model)
    
    def forward(self, x):
        x_temp = x[:,:,4:]

        
        return self.embed(x_temp)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.temporal_embedding_nn = TemporalEmbedding(d_model = d_model)
        self.temporal_embedding_time = TimeFeatureEmbedding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        value_embed = self.value_embedding(x)
        position_embed = self.position_embedding(x)
        temporal_embed = self.temporal_embedding_nn(x_mark)
        time_embed = self.temporal_embedding_time(x_mark)

        x = value_embed + position_embed + temporal_embed + time_embed
        
        return self.dropout(x)
