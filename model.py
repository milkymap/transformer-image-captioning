import numpy as np 
import torch as th 
import torch.nn as nn 

from core import Transformer, PositionalEncoding

class CaptionTransformer(nn.Module):
    def __init__(self, in_dim, hd_dim, ff_dim, nb_heads, num_encoders, num_decoders, pre_norm, seq_length, nb_tokens, padding_idx):
        super(CaptionTransformer, self).__init__()
        self.embedding_scale = np.sqrt(hd_dim)
        self.position_encoder = PositionalEncoding(seq_length, hd_dim)
        self.adaptaror = nn.Linear(in_dim, hd_dim)
        self.token_embedder = nn.Embedding(nb_tokens, hd_dim, padding_idx) 
        self.transformer = Transformer(
            in_dim=hd_dim,
            ff_dim=ff_dim,
            nb_heads=nb_heads,
            encoder_depth=num_encoders,
            decoder_depth=num_decoders,
            pre_norm=pre_norm
        )
        self.generator = nn.Linear(hd_dim, nb_tokens)
    

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.adaptaror(src)  # reduce the dimension for in_dim to hd_dim 
        src = self.position_encoder(src)
        memory = self.transformer.encoder(src, src_mask, src_key_padding_mask)
        return memory 
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.token_embedder(tgt) * self.embedding_scale
        tgt = self.position_encoder(tgt)
        output = self.transformer.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.position_encoder(self.adaptaror(src))
        embedded_tgt = self.token_embedder(tgt)
        embedded_tgt = self.position_encoder(embedded_tgt)

        output = self.transformer(
            src=src, 
            tgt=embedded_tgt, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return self.generator(output[-1])

         
