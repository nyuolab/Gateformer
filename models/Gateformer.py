import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PatchEmbedding, DataEmbedding_inverted


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        patch_len = configs.patch_len
        stride = configs.stride
        padding = stride

        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.d_model, head_dropout=configs.dropout)
        self.project_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.gate_w1 = nn.Linear(configs.d_model, configs.d_model)
        self.gate_w2 = nn.Linear(configs.d_model, configs.d_model)
        self.gate_w3 = nn.Linear(configs.d_model, configs.d_model)
        self.gate_w4 = nn.Linear(configs.d_model, configs.d_model)
        self.gate_sigmoid = nn.Sigmoid()

        enc_layers_temporal = nn.TransformerEncoderLayer(
                                    d_model=configs.d_model,
                                    nhead=configs.n_heads,
                                    dim_feedforward=configs.d_ff,
                                    dropout=configs.dropout,
                                    layer_norm_eps=1e-5,
                                    batch_first=True, 
                                    norm_first=True 
                                )

        enc_layers_feature = nn.TransformerEncoderLayer(
                                    d_model=configs.d_model,
                                    nhead=configs.n_heads,
                                    dim_feedforward=configs.d_ff,
                                    dropout=configs.dropout,
                                    layer_norm_eps=1e-5,
                                    batch_first=True,
                                    norm_first=True
                                )
        
        # temporal-wise attention
        self.enc_temporal = nn.TransformerEncoder(enc_layers_temporal, num_layers=configs.e_layers)

        # variate-wise attention
        self.enc_variate = nn.TransformerEncoder(enc_layers_feature, num_layers=configs.e_layers)

        # output projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        # x_enc: [B, L, D]
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        global_embedding = self.project_embedding(x_enc, None) # global_embedding: [B, D, d_model]

        # patching
        x_enc = x_enc.permute(0, 2, 1) 
        enc_out, n_vars = self.patch_embedding(x_enc) # enc_out: [B * D, patch_num, d_model]

        # temporal-wise attention
        enc_out = self.enc_temporal(enc_out) # enc_out: [B * D, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        temporal_dependency_embedding = self.head(enc_out) # temporal_dependency_embedding: [B, D, d_model]

        # gate
        gate = self.gate_sigmoid(self.gate_w1(global_embedding) + self.gate_w2(temporal_dependency_embedding))
        enc_out = gate * global_embedding + (1 - gate) * temporal_dependency_embedding

        # variate-wise attention
        enc_out_cross = self.enc_variate(enc_out)

        # gate
        gate = self.gate_sigmoid(self.gate_w3(enc_out) + self.gate_w4(enc_out_cross))
        enc_out = gate * enc_out + (1 - gate) * enc_out_cross

        # Decoder
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]  # dec_out: [B, pred_len, D]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]