import torch.nn as nn
import torch
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # print(f"x dtype - shape: {x.dtype} - {x.shape}" , f" - dmodel: {type(self.d_model)}")
        # print("x: ", x)

        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_leng: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_leng = seq_leng
        self.dropout = nn.Dropout(dropout)

        # create the main positional matrix (seq_leng * d_model)
        pe = torch.zeros(seq_leng, d_model)
        # create position vector of shape (seq_leng, 1)
        pos = torch.arange(0, seq_leng, dtype=torch.float).unsqueeze(1)
        # create odd and even indexes
        odd_indx = torch.arange(1, d_model, 2)
        even_indx = torch.arange(0, d_model, 2)
        # calculate the shared term
        div_term = torch.exp(even_indx.unsqueeze(0).float() * (-1 / d_model) * math.log(10000))
        # create the matrix of (seq_leng * d_model/2)
        param = pos * div_term
        # put sin and cosin of the param in the right position of the pe.
        pe[:, even_indx] = torch.sin(param)
        pe[:, odd_indx] = torch.cos(param)

        # to capture the batches
        pe = pe.unsqueeze(0)
        # print("pe shape: ", pe.shape)
        # to register pe as the fixed matrix
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x seems to be in the shape of (batch, sentence_length, d_model)
        try:
            x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        except Exception as e:
            raise Exception(f"pe shape: {self.pe.shape} - x shape: {x.shape} - error: {e}")
        return self.dropout(x)


class NormalizationLayer(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiply
        self.bias = nn.Parameter(torch.zeros(1)) # Add

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x- mean)/(std + self.eps) + self.bias



class FeedForward(nn.Module):
    def __init__(self, in_features: int, d_f: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, d_f)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear_2 = nn.Linear(d_f, in_features)

    def forward(self, x):
        # (batch, seq_length, d_model) --> (batch, seq_length, d_f) ---> (batch, seq_length, d_model)
        # We might be able to change this FF unit with VAE encoder/decoders
        out1 = self.linear_1(x)
        out2 = self.dropout(self.relu(out1))
        out = self.linear_2(out2)
        return out


class MultiHeadAttention(nn.Module):
    # considers relation of the words within the sentence they occur
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.heads = heads

        self.w_q = nn.Linear(d_model, d_model) # x * w_q
        self.w_k = nn.Linear(d_model, d_model) # x * w_k^T
        self.w_v = nn.Linear(d_model, d_model) # x * w_v^T
        self.w_o = nn.Linear(d_model, d_model)
        #
        assert d_model%heads == 0
        self.d_k = d_model // heads
        self.dropout = nn.Dropout(dropout)

    def attention(self, Q, K, V, mask= None, dropout = nn.Dropout(0.01)):
        attention_scors = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
        if mask != None:
            attention_scors.masked_fill_(mask == 0, -1e9)

        attention_scors = attention_scors.softmax(dim = -1)
        if dropout:
            attention_scors = dropout(attention_scors)
        return attention_scors @ V, attention_scors

    def forward(self, q, k, v, mask = None):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # heads
        Q = Q.view(Q.shape[0], Q.shape[1], self.heads, self.d_k).transpose(1,2)
        K = K.view(K.shape[0], K.shape[1], self.heads, self.d_k).transpose(1,2)
        V = V.view(V.shape[0], V.shape[1], self.heads, self.d_k).transpose(1,2)

        heads, self.attention_scores = self.attention(Q, K, V, mask, self.dropout)
        # print(f"heads shape: {heads.shape}")
        heads = heads.transpose(1,2)
        heads = heads.contiguous()
        heads = heads.view(heads.shape[0], -1, self.heads * self.d_k)
        # print(f"final heads shape: {heads.shape}")


        return self.w_o(heads)

    # def forward_(self, q, k, v, mask):
    #     Q = self.w_q(q)
    #     K = self.w_k(k)
    #     V = self.w_v(v)
    #
    #     # heads
    #     Heads = []
    #     for h in range(self.heads):
    #         Heads.append([Q[:,:,h*self.d_k : (h+1)*self.d_k], K[:,:,h*self.d_k : (h+1)*self.d_k], V[:,:,h*self.d_k : (h+1)*self.d_k]])
    #
    #     Attention_Heads = []
    #     x_Heads = []
    #     for h in  range(self.heads):
    #         x_head, att_head = self.attention(mask, self.dropout, **Heads[h])
    #         x_Heads.append(x_head)   # each processed head has (batch, seq_length, d_k)
    #         Attention_Heads.append(att_head)   # each processed head has (batch, seq_length, d_k)
    #
    #     x_cat = torch.cat(x_Heads, dim=-1)  # (batch, seq_length, d_model)
    #     return x_cat * self.w_o



class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = NormalizationLayer()

    def forward(self, x, sublayer):
        # return x + self.dropout(self.norm(sublayer(x))) # might want to substitute
        return x + self.dropout(self.norm(sublayer(x)))


class EncoderUnit(nn.Module):
    def __init__(self, d_model, d_f, heads, dropout):
        super().__init__()
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, d_f)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x,x,x,src_mask) )
        x = self.residual_connection[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_n: int, d_model: int, d_f: int, heads: int, dropout: float):
        super().__init__()
        self.encoder_n = encoder_n
        self.encoders = nn.ModuleList([EncoderUnit(d_model, d_f, heads, dropout) for _ in range(encoder_n)])
        self.norm = NormalizationLayer()
    def forward(self, x, src_mask):
        first_encoder = True
        for _ in range(self.encoder_n):
            if(first_encoder):
                x = self.encoders[_](x, src_mask)
                first_encoder = False

            x = self.encoders[_](x, None)

        # return self.norm(x)   # might want to substitute - however, I guess this might change the nature of input is going to be fed to the next encoder layer.
        return x


class DecoderUnit(nn.Module):
    def __init__(self, d_model, d_f, heads, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
        self.encoder_integrate_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, d_f)

    def forward(self, x, x_enc, dec_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, dec_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention(x, x_enc, x_enc, target_mask))
        x = self.residual_connections[2](x, self.feed_forward)

        return x

class Decoder(nn.Module):
    def __init__(self, decoder_n: int, d_model: int, d_f: int, heads: int, dropout: float):
        super().__init__()
        self.decoder_n = decoder_n
        self.decoders = nn.ModuleList(DecoderUnit(d_model, d_f, heads, dropout) for _ in range(decoder_n))
        self.norm = NormalizationLayer()
    def forward(self, x, x_enc, dec_mask, target_mask):
        for i in range(self.decoder_n):
            x = self.decoders[i](x, x_enc, dec_mask, target_mask)

        # return self.norm(x)  # might want to substitute - however, I guess this might change the nature of input is going to be fed to the next decoder layer.
        return x



class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        vocab_proj = self.layer(x)

        return vocab_proj       # if apply CrossEntropyLoss
        # return torch.log_softmax(vocab_proj, dim= -1)




class Transformer(nn.Module):
    def __init__(self, encoder_n: int, decoder_n: int, d_model: int, d_f: int, heads: int, src_vocab: int, tgt_vocab: int, src_seq: int, tgt_seq: int, dropout: float):
        super().__init__()
        self.src_embedding = InputEmbeddings(d_model, src_vocab)
        self.src_position = PositionalEncoding(d_model, src_seq, dropout)
        self.tgt_embedding = InputEmbeddings(d_model, tgt_vocab)
        self.tgt_position = PositionalEncoding(d_model, tgt_seq, dropout)

        self.encoders = Encoder(encoder_n, d_model, d_f, heads, dropout)
        self.decoders = Decoder(decoder_n, d_model, d_f, heads, dropout)

        self.projection_layer = ProjectionLayer(d_model, tgt_vocab)

    def encode(self, x, src_mask):
        # (batch, seq, d_model)
        # print("- embedding")
        x = self.src_embedding(x)
        # print("- positioning")
        x = self.src_position(x)
        # print("- encoder")
        x = self.encoders(x, src_mask)

        # you might want to apply norm here
        return x

    def decode(self, x, x_enc, enc_mask, tgt_mask):

        x = self.tgt_embedding(x)
        x = self.tgt_position(x)
        x = self.decoders(x, x_enc, enc_mask, tgt_mask)

        return x

    def project(self, x):
        return self.projection_layer(x)



def build_transformer(src_vocab: int, tgt_vocab: int, src_seq: int, tgt_seq: int, dropout: float = 0.1, N_enc: int = 6, N_dec: int = 6, d_model: int = 512, d_f: int= 2048, heads: int = 8):
    transformer = Transformer(N_enc, N_dec, d_model, d_f, heads, src_vocab, tgt_vocab,src_seq, tgt_seq, dropout)

    for p in transformer.parameters():
        if(p.dim() > 1):
            nn.init.xavier_uniform_(p)

    return transformer



def get_model(config, src_vocab, tgt_vocab):

    transformer_model = build_transformer(src_vocab= src_vocab, tgt_vocab= tgt_vocab, src_seq= config["seq_length"], tgt_seq= config["seq_length"], dropout= config["model"]["dropout"], d_model = config["model"]["d_model"], d_f = config["model"]["d_f"], heads= config["model"]["heads"])

    return transformer_model



