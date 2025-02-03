import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer, Siamese_DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from utils.tools import ContrastiveWeight, AggregationRebuild, patch_random_masking
from utils.losses import AutomaticWeightedLoss

class Flatten_Head(nn.Module):
    def __init__(self, nf, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x patch_num x d_model]
        x = self.flatten(x) # [bs x n_vars x (patch_num * d_model)]
        x = self.linear(x) # [bs x n_vars x pred_len]
        x = self.dropout(x) # [bs x n_vars x pred_len]
        return x


class Pooler_Head(nn.Module):
    def __init__(self, patch_num, d_model, head_dropout=0):
        super().__init__()

        pn = patch_num * d_model
        print(patch_num, d_model, pn)
        dimension = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [bs x n_vars x patch_num x d_model]
        x = self.pooler(x)  # [bs x n_vars x dimension]
        return x
    
    
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
    
    
def generate_continuous_mask(B, T, C=None, n=5, l=0.1):
    if C:
        res = torch.full((B, T, C), True, dtype=torch.bool)
    else:
        res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            if C:
                # For a continuous timestamps, mask random half channels
                index = np.random.choice(C, int(C/2), replace=False)
                res[i, t:t + l, index] = False
            else:
                # For a continuous timestamps, mask all channels
                res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, C=None, p=0.5):
    if C:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T, C))).to(torch.bool)
    else:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        # projection head for finetune
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

        self.repr_dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.repr_dropout(self.proj_head(x))
        if self.output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x


class FTClassifier(nn.Module):
    def __init__(self, input_dims, output_dims, depth, p_output_dims, hidden_dims=64, p_hidden_dims=128,
                 device='cuda', multi_gpu=True):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.p_hidden_dims = p_hidden_dims  # Cph
        self.p_output_dims = p_output_dims  # Cp
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        # projection head for finetune
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        device = torch.device(device)
        if device == torch.device('cuda') and multi_gpu:
            self._net = nn.DataParallel(self._net)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net.to(device)
        self.proj_head.to(device)

        # stochastic weight averaging, see link:
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)


    def forward(self, x):
        out = self.net(x)  # B x O x Co
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).transpose(1, 2)  # B x 1 x Co
        out = out.squeeze(1)  # B x Co
        x = self.proj_head(out)  # B x Cp
        if self.p_output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='all_true'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, x, mask=None, pool=False):  # input dimension : B x O x Ci
        x = self.input_fc(x)  # B x O x Ch (hidden_dims)
        
        # generate & apply mask, default is binomial
        if mask is None:
            # mask should only use in training phase
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')

        # mask &= nan_masK
        # ~ works as operator.invert
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x O
        
        if pool:
            x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        else:
            x = x.transpose(1, 2)  # B x O x Co
        
        return x
    
    
class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = configs.stride
        self.lineage_tokens = configs.lineage_tokens

        self.configs = configs

        self.visual = 0
        self.visual_path = None

        # patching and embedding
        # self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        # Temporal shift token
        if self.lineage_tokens:
            print("init {} lineage tokens!".format(self.lineage_tokens))

            if self.configs.current_token:
                self.token_0 = nn.Parameter(torch.zeros(1, 1, configs.d_model), requires_grad=True) # current lineage token

            for i in range(self.lineage_tokens):
                setattr(self, f'token_{i + 1}', nn.Parameter(torch.zeros(1, 1, configs.d_model), requires_grad=True)) # past lineage token
            self.initialize_weights()

        mask_flag = False

        # Encoder
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(mask_flag, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        self.encoder = TSEncoder(depth=configs.e_layers, input_dims=configs.d_model, output_dims=configs.d_model)

        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 2)
        self.head_nf = self.patch_num * configs.d_model

        if self.task_name == 'patchtst':
            self.projection = nn.Linear(configs.d_model, configs.patch_len, bias=True) # decoder to patch
        elif self.task_name == 'simmtm':

            # for series-wise representation
            self.pooler = Pooler_Head(self.patch_num, configs.d_model, head_dropout=configs.head_dropout)

            # for reconstruction
            self.projection = Flatten_Head(self.head_nf, configs.seq_len, head_dropout=configs.head_dropout)


            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(self.configs)
            self.aggregation = AggregationRebuild(self.configs)
            self.mse = torch.nn.MSELoss()

        elif self.task_name == 'timesiam':

            self.decoder = Decoder(
                [
                    Siamese_DecoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

            self.projection = Flatten_Head(self.head_nf, configs.seq_len, head_dropout=configs.head_dropout)

        elif self.task_name in ['linear_probe', 'fine_tune', 'fine_tune_part']:
            self.representation_using = configs.representation_using

            if self.lineage_tokens and self.representation_using == 'concat':
                print("{}>  Representation ({}), head dimension {}*{}".format('-'*50, self.representation_using, configs.d_model, self.lineage_tokens))
                self.head = Flatten_Head(self.head_nf*self.lineage_tokens, configs.pred_len, head_dropout=configs.head_dropout)
            else:
                print("{}> Representation ({}), head dimension {}".format('-'*50, self.representation_using, configs.d_model))
                self.head = Flatten_Head(self.head_nf, configs.pred_len, head_dropout=configs.head_dropout)
        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = Flatten_Head(self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = Flatten_Head(self.head_nf, configs.seq_len, head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def initialize_weights(self):
        for i in range(self.lineage_tokens):
            token = getattr(self, f'token_{i + 1}')
            torch.nn.init.normal_(token, std=.02)

    def norm(self, x, mask=None, means=None, stdev=None):

        if means is not None and stdev is not None:
            x = x - means
            x /= stdev
        else:
            if mask is not None:
                means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1)
                means = means.unsqueeze(1).detach()
                x = x - means
                x = x.masked_fill(mask == 0, 0)
                stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
                stdev = stdev.unsqueeze(1).detach()
                x /= stdev
            else:
                means = x.mean(1, keepdim=True).detach()
                x = x - means
                stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x /= stdev

        return x, means, stdev

    def denorm(self, x, means, stdev, L):
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return x

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars, _ = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars, _ = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars, _ = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars, _ = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def patch_mask_loss(self, preds, target, mask):
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len]
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def pretrain_patchtst(self, x_enc):

        # Normalization from Non-stationary Transformer
        x_enc, means, stdev = self.norm(x_enc) # [bs x seq_len x n_vars]

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)

        bs, n_vars, seq_len = x_enc.shape

        # patchify
        x_enc = self.patch_embedding.patchify(x_enc) # [(bs * n_vars) x num_patch x patch_len]
        x_enc = x_enc.reshape(-1, n_vars, x_enc.shape[-2],  x_enc.shape[-1])
        x_enc = x_enc.permute(0, 2, 1, 3) # [bs x num_patch x n_vars x patch_len]

        # patch masking
        x_mask, _, mask, _ = patch_random_masking(x_enc, self.configs.mask_rate)   # xb_mask: [bs x num_patch x n_vars x patch_len]  mask [bs x num_patch x n_vars]
        mask = mask.bool()
        x_ = x_mask.permute(0, 2, 1, 3) # [bs x n_vars x num_patch x patch_len]
        x_ = x_.reshape(-1, x_.shape[-2], x_.shape[-1]) # [(bs*n_vars) x num_patch x patch_len]

        # value embedding
        x_ = self.patch_embedding.value_embedding(x_) # [(bs*n_vars) x num_patch x d_model]

        # position embedding
        x_ = self.patch_embedding.dropout(x_ + self.patch_embedding.position_embedding(x_)) # [(bs*n_vars) x num_patch x d_model]

        # Encoder
        enc_out, _ = self.encoder(x_) # [(bs*n_vars) x patch_num x d_model]

        # prediction
        pred = self.projection(enc_out)  # [(bs*n_vars) x patch_num x patch_len]
        pred = pred.reshape(bs, n_vars, pred.shape[-2], pred.shape[-1]) # [bs x n_vars x num_patch x patch_len]
        pred = pred.permute(0, 2, 1, 3)  # [bs x num_patch x n_vars x patch_len]

        # loss
        loss = self.patch_mask_loss(pred, x_enc, mask)

        return loss, mask, pred, x_enc

    def pretrain_simmtm(self, x_enc, x_mark_enc, batch_x, mask):

        bs, seq_len, n_vars = x_enc.shape

        # Normalization
        x_enc, means, stdev = self.norm(x_enc, mask)

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars, _ = self.patch_embedding(x_enc) # [(bs * n_vars) x patch_num x d_model]

        # Encoder
        p_enc_out, attns = self.encoder(enc_out) # [(bs * n_vars) x patch_num x d_model]

        # series-wise representation
        s_enc_out = self.pooler(p_enc_out) # [(bs * n_vars) x dimension]

        # series weight learning
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out) # similarity_matrix: [(bs * n_vars) x (bs * n_vars)]
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out) # agg_enc_out: [(bs * n_vars) x patch_num x d_model]

        agg_enc_out = agg_enc_out.reshape(bs, n_vars, agg_enc_out.shape[-2], agg_enc_out.shape[-1]) # agg_enc_out: [bs x n_vars x patch_num x d_model]

        # Decoder
        dec_out = self.projection(agg_enc_out)  # [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs x seq_len x n_vars]

        # De-Normalization
        self.pred_len = self.seq_len
        dec_out = self.denorm(dec_out, means, stdev)

        pred_batch_x = dec_out[:batch_x.shape[0]]

        # series reconstruction
        loss_rb = self.mse(pred_batch_x, batch_x.detach())

        # loss
        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x

    def pretrain_timesiam(self, past_x_enc, cur_x_enc, segment, mask):

        bs, seq_len, n_vars = past_x_enc.shape

        # Normalization
        past_x_enc, _, _ = self.norm(past_x_enc) # [bs x seq_len x n_vars]
        cur_x_enc, means, stdev = self.norm(cur_x_enc, mask=mask) # [bs x seq_len x n_vars]

        """past window"""
        # past window patch embedding
        past_x_enc = past_x_enc.permute(0, 2, 1)  # past_x_enc: [bs x n_vars x seq_len]
        past_x_enc, n_vars, _ = self.patch_embedding(past_x_enc) # past_x_enc: [(bs*n_vars) x patch_num x d_model]

        # add past temporal shift token
        segment = segment.repeat_interleave(n_vars)

        if self.lineage_tokens:
            selected_tensors = []
            for i in segment:
                token = getattr(self, f'token_{i + 1}')
                selected_tensors.append(token)
            lineages = torch.cat(selected_tensors, dim=0)
            lineages = lineages.repeat(1, past_x_enc.shape[1], 1).to(past_x_enc.device)  # [(bs*n_vars)  x patch_num x d_model]
            past_x_enc = past_x_enc + lineages  # past_x_enc: [(bs*n_vars)  x patch_num x d_model]

        # encoder - past window
        past_enc_out, attns = self.encoder(past_x_enc, attn_mask=None) # past_enc_out: # [(bs*n_vars)  x patch_num x d_model]

        """current window"""
        # current window patch embedding
        cur_x_enc = cur_x_enc.permute(0, 2, 1) # cur_x_enc: [bs x n_vars x seq_len]
        cur_x_enc, n_vars, _ = self.patch_embedding(cur_x_enc)  # cur_x_enc: [(bs*n_vars) x patch_num x patch_len]

        # add current temporal shift token
        if self.lineage_tokens and self.configs.current_token:
            cur_x_enc = cur_x_enc + self.token_0.repeat(cur_x_enc.shape[0], cur_x_enc.shape[1], 1).to(cur_x_enc.device)  # cur_x_enc: [(bs*n_vars)  x patch_num x d_model]

        # encoder
        cur_enc_out, attns = self.encoder(cur_x_enc, attn_mask=None) # cur_enc_out: [(bs*n_vars) x patch_num x d_model]

        """cross past and current window"""
        # decoder
        dec_out = self.decoder(cur_enc_out, past_enc_out) # dec_out: [(bs*n_vars)  x patch_num x d_model]
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]) # dec_out: [bs x n_vars x patch_num x d_model]

        pred = self.projection(dec_out) # [bs x n_vars x seq_len]
        pred = pred.permute(0, 2, 1)

        # De-Normalization
        pred = self.denorm(pred, means, stdev, self.seq_len) # [bs x seq_len x n_vars]

        return pred

    def temperal_shift_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        bs, seq_len, n_vars = x_enc.shape   # [bs x seq_len x n_vars]

        # Normalization from Non-stationary Transformer
        x_enc, means, stdev = self.norm(x_enc) # [bs x seq_len x n_vars]

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_emb, n_vars, _ = self.patch_embedding(x_enc) # [bs * n_vars x patch_num x d_model]

        # add temperal shift token
        if self.lineage_tokens and self.lineage_tokens != 0:

            if self.representation_using == 'concat':
                outputs = []
                for i in range(self.lineage_tokens):

                    # get and add temperal shift token
                    token = getattr(self, f'token_{i + 1}')
                    enc_input = enc_emb + token.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)

                    # encoder
                    enc_out, attns = self.encoder(enc_input, attn_mask=None) # [(bs*n_vars) x patch_num x d_model]

                    outputs.append(enc_out)
                enc_out = torch.cat(outputs, dim=-1) # [(bs*n_vars) x patch_num x (d_model*tokens)]
            elif self.representation_using == 'avg':
                
                outputs = []

                # current token
                if self.configs.current_token:
                    enc_input = enc_emb + self.token_0.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)

                    enc_out, attns = self.encoder(enc_input, attn_mask=None)  # [(bs*n_vars) x patch_num x d_model]
                    outputs.append(enc_out)

                # past token
                for i in range(self.lineage_tokens):

                    # get and add temperal shift token
                    token = getattr(self, f'token_{i + 1}')
                    enc_input = enc_emb + token.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)

                    # encoder
                    enc_out, attns = self.encoder(enc_input, attn_mask=None) # [(bs*n_vars) x patch_num x d_model]
                    outputs.append(enc_out)

                enc_out = torch.stack(outputs).mean(dim=0) # [(bs*n_vars) x patch_num x d_model]
        else:
            # encoder
            enc_out, attns = self.encoder(enc_emb, attn_mask=None)

        dec_out = self.head(enc_out) # [bs * n_vars x pred_len]
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-1]) # [bs x n_vars x pred_len]
        dec_out = dec_out.permute(0, 2, 1) # [bs x pred_len x n_vars]

        # De-Normalization from Non-stationary Transformer
        dec_out = self.denorm(dec_out, means, stdev, self.pred_len) # [bs x pred_len x n_vars]
        return dec_out

    def temperal_extend_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        bs, seq_len, n_vars = x_enc.shape   # [bs x seq_len x n_vars]

        # Normalization from Non-stationary Transformer
        x_enc, means, stdev = self.norm(x_enc) # [bs x seq_len x n_vars]

        new_seq_len = seq_len // self.lineage_tokens  # equal pre-training sequence length

        x_enc = x_enc.reshape(bs, self.lineage_tokens, new_seq_len, n_vars) # [bs x n x new_seq_len x n_vars]

        outputs = []
        for i in range(self.lineage_tokens):

            new_x_enc = x_enc[:, i, :, :] # [bs x new_seq_len x n_vars]

            # do patching and embedding
            new_x_enc = new_x_enc.permute(0, 2, 1) # [bs x n_vars x new_seq_len]
            enc_emb, n_vars, _ = self.patch_embedding(new_x_enc) # [(bs * n_vars) x patch_num x d_model]

            # get and add temperal shift token
            token = getattr(self, f'token_{i + 1}')
            enc_input = enc_emb + token.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)

            # encoder
            enc_out, attns = self.encoder(enc_input, attn_mask=None) # [(bs * n_vars) x patch_num x d_model]

            outputs.append(enc_out)

        enc_out = torch.cat(outputs, dim=-1) # [(bs * n_vars) x patch_num x (n * d_model)]

        dec_out = self.head(enc_out) # [(bs * n_vars) x pred_len]
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-1]) # [bs x n_vars x pred_len]
        dec_out = dec_out.permute(0, 2, 1) # [bs x pred_len x n_vars]

        # De-Normalization from Non-stationary Transformer
        dec_out = self.denorm(dec_out, means, stdev, self.pred_len) # [bs x pred_len x n_vars]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, segment=None, mask=None, t=None, noise=None):
        if self.task_name == 'patchtst':
            return self.pretrain_patchtst(x_enc)
        if self.task_name == 'simmtm':
            return self.pretrain_simmtm(x_enc, x_mark_enc, x_dec, mask)
        if self.task_name == 'timesiam':
            return self.pretrain_timesiam(x_enc, x_dec, segment=segment, mask=mask)
        if  self.task_name in ['linear_probe', 'fine_tune', 'fine_tune_part']:
            return self.temperal_shift_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None