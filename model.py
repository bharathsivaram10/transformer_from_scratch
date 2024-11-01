# This all follows the video https://www.youtube.com/watch?v=ISNdQcPhsts&list=WL&index=10
# Some extra comments were added for clarity
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
# Uses the same amount of dimensions as the input embedding
# Will be added to Input embedding so the resultant embedding carries info about the mearning AND position
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # According to the sine and cosine functions in Attention paper
        # Matrix of shape (seq_len, d_model)
        PE = torch.zeros(seq_len, d_model)

        # Create vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0) / d_model))
        
        # For even numbered dimensions
        PE[:, 0::2] = torch.sin(position * div_term)

        # For odd numbered dimensions
        PE[:,1::2] = torch.cos(position * div_term)

        # Adding a batch dimension
        PE = PE.unsqueeze(0) # (1, seq_len, d_model)

        # Tensor will be saved
        self.register_buffer('PE',PE)

    def forward(self,x):

        # We need to PE to match the sequence length of input hence x.shape[1]
        # Don't need to backprop, this is a constant tensor, not learned
        x = x + (self.PE[:, :x.shape[1],:]).requires_grad_(False)

        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Mulitplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self,x):

        # Everything after the batch, hence -1
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model : int, d_ff: int, dropout: float ):
        super().__init__()
        self.ln1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ln2(x)

        return x
    
class MultiHeadAttentionBlock(nn.Module):

    # Model dimensions, # of heads, dropout prob
    def __init__(self, d_model:int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        # Need to divide the embedding vector into h heads, so d_model  must be divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model / h
        self.d_v = d_model / h


        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv

        self.w_o = nn.Linear(d_model,d_model) # wo
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]

        # (B, h, seq_len, d_k) @ (B, h, d_k, seq_len) --> (B, h, seq_len, seq_len)
        # The last dim of matrix A must match second to last dim of matrix B
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):

        query = self.w_q(q) # (B, seq_len, d_model) --> (B, seq_len, d_model)
        key = self.w_k(k) # (B, seq_len, d_model) --> (B, seq_len, d_model)
        value = self.w_v(v) # (B, seq_len, d_model) --> (B, seq_len, d_model)

        # (B, seq_len, d_model) --> (B, seq_len, h, d_k) --> (B, h, seq_len, d_k)
        # We do transpose so that we can select by the head dimension and can do matmul with the
        # (seq_len, d_k) portion
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (B, h, seq_len, d_k) -> (B, seq_len, h, d_k) --> (B, seq_len, d_model)
        # We use contiguous b/c the transpose means the tensor's data in memory isn't contiguous
        # wrt to its new layout. So contigous rearranged the memory to match the new logical view. 
        # Then we can call view
        x = x.transpose(1,2).contiguous().view(x.shape[0],x.shape[1],self.h * self.d_k)

        # (B, seq_len, d_model) @ (d_model, d_model) --> (B, seq_len, d_model)
        x = x @ self.w_o

        return x
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):

        # The paper actually applies sublayer first then does norm

        # Add & Norm
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)

        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, ff_block : FeedForwardBlock, dropout: nn.Dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout)] for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[1](x, self.ff_block)

        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self,x):
        # (B, seq_len, d_model) --> (B, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff : float = 2048):
    
    # Create embedding layers
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)

    # Position encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create N encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create N Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, ff_block)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create Projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    # Initialize Params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer