import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
import random

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor, d_model, n_heads, e_layers, d_layers, d_ff, 
                dropout, attn, activation, 
                output_attention, distil=True, mix=True, 
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding embed는 timeF가 지금 default임.
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)#d_model에서 c_out으로 가는거 생각해.
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        여기 지금 enc_self_mask, dec_self_mask, dec_enc_mask None으로 돼 있는 거 체크해야함.
        """
        # print(f"x_enc shape : {x_enc.shape}, x_mark_enc shape : {x_mark_enc.shape}, x_dec shape : {x_dec.shape}, x_mark_dec shape : {x_mark_dec.shape}")
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print(f"enc_out(enc_embedding 통과) shape : {enc_out.shape}")
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(f"enc_out(encoder 통과) shape : {enc_out.shape}")

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # print(f"dec out(decoder embedding 통과) shape : {dec_out.shape}")
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # print(f"dec_out(decoder 통과) shape : {dec_out.shape}")
        dec_out = self.projection(dec_out)
        # print(f"dec out(final projection 통과) shape : {dec_out.shape}")
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        """
        error code
        """
        # self.pred_len = random.randint(5, 2)
        # self.pred_len = 20
        # print("pred_len")
        # print(self.pred_len)
        """
        error code end
        """
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]