import torch
import torch.nn as nn
from block import get_sinusoid_encoding_table, get_attn_key_pad_mask, get_non_pad_mask, \
    get_subsequent_mask, EncoderLayer
from config import PAD, KS, Fea_PLUS


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        # -- Forward
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device,
            d_feature, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            class_num=5):

        super().__init__()

        self.encoder = Encoder(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device

        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model+Fea_PLUS, class_num)
        # try different linear style
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)

    def forward(self, src_seq, fea_plus):
        b, l = src_seq.size()
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output = enc_output
        res = self.linear1_cov(dec_output)
        res = res.contiguous().view(res.size()[0], -1)
        res = torch.cat((res,fea_plus),1)
        res = self.linear1_linear(res)
        return res