import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm.auto import tqdm


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, dropout=0.1):
    super().__init__()
    self.query = nn.Linear(embed_dim, embed_dim)
    self.key = nn.Linear(embed_dim, embed_dim)
    self.value = nn.Linear(embed_dim, embed_dim)
    self.proj = nn.Linear(embed_dim, embed_dim)

    self.attn_drop = nn.Dropout(dropout)

    self.n_head = num_heads
    self.emd_dim = embed_dim
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** 0.5

  def kv_cache_forward(self, query, key, value, kv_cache=None):
    N, S, D = query.shape
    N, T, D = value.shape
    H = self.n_head
    D_H = self.head_dim

    query_i = query[:, -1:]
    key_i = key[:, -1:]
    value_i = value[:, -1:]

    Q_i = self.query(query_i).view(N, 1, H, D_H).permute(0, 2, 1, 3)
    K_i_T = self.key(key_i).view(N, 1, H, D_H).permute(0, 2, 3, 1)
    V_i = self.value(value_i).view(N, 1, H, D_H).permute(0, 2, 1, 3)

    if 'K' in kv_cache and 'V' in kv_cache:
      K_lt_i_T = kv_cache['K']
      V_lt_i = kv_cache['V']
      K_T = torch.cat((K_lt_i_T, K_i_T), dim=-1)
      V = torch.cat((V_lt_i, V_i), dim=-2)
    else:
      K_T = K_i_T
      V = V_i
    # update kv_cache
    kv_cache['K'] = K_T
    kv_cache['V'] = V

    attn_scores = Q_i @ K_T / self.scale

    output = self.attn_drop(torch.softmax(attn_scores, dim=-1)) @ V
    output = output.permute(0, 2, 1, 3).reshape(N, 1, D)
    output = self.proj(output)
    return output

  def forward(self, query, key, value, attn_mask=None, kv_cache=None):
    if kv_cache is not None:
      return self.kv_cache_forward(query, key, value, kv_cache)

    N, S, D = query.shape
    N, T, D = value.shape
    H = self.n_head
    D_H = self.head_dim
    
    Q = self.query(query).view(N, S, H, D_H).permute(0, 2, 1, 3)
    K_T = self.key(key).view(N, T, H, D_H).permute(0, 2, 3, 1)
    V = self.value(value).view(N, T, H, D_H).permute(0, 2, 1, 3)

    attn_scores = Q @ K_T / self.scale
    if attn_mask is not None:
      attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e10)
    
    # (N, H, S, T) @ (N, H, T, D/H) -> (N, H, S, D/H)
    output = self.attn_drop(torch.softmax(attn_scores, dim=-1)) @ V
    # (N, H, S, D/H) -> (N, S, D)
    output = output.permute(0, 2, 1, 3).reshape(N, S, D)
    output = self.proj(output)
    return output


class FeedForwardNetwork(nn.Module):
  def __init__(self, embed_dim, ffn_dim, dropout=0.1):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(embed_dim, ffn_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(ffn_dim, embed_dim)
    )

  def forward(self, x):
    return self.layers(x)


class CausalTransformerDecoderLayer(nn.Module):
  def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
    super().__init__()
    self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
    self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

    self.norm_self = nn.LayerNorm(input_dim)
    self.norm_ffn = nn.LayerNorm(input_dim)

    self.dropout_self = nn.Dropout(dropout)
    self.dropout_ffn = nn.Dropout(dropout)

  def forward(self, x, mask=None, kv_cache=None):
    shortcut = x
    x = self.self_attn(query=x, key=x, value=x, attn_mask=mask, kv_cache=kv_cache)
    x = self.dropout_self(x)
    x = x + shortcut
    x = self.norm_self(x)

    shortcut = x
    x = self.ffn(x)
    x = self.dropout_ffn(x)
    x = x + shortcut
    x = self.norm_ffn(x)

    return x


def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CausalTransformerDecoder(nn.Module):
  def __init__(self, decoder_layer, num_layers):
    super().__init__()
    self.layers = clones(decoder_layer, num_layers)
    self.num_layers = num_layers

  def forward(self, x, mask=None, kv_cache=None):
    output = x
    for mod in self.layers:
      output = mod(output, mask=mask, kv_cache=kv_cache)
    return output


class CausalTransformer(nn.Module):
  def __init__(self, vocab_size, dim_model, num_heads, num_layers,
               dim_feedforward, max_length, dropout=0.1):
    super().__init__()
    self.vocab_size = vocab_size
    self.max_length = max_length

    self.embedding = nn.Embedding(vocab_size, dim_model)
    self.positional_encoding = nn.Parameter(torch.randn(max_length, dim_model))

    decoder_layer = CausalTransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
    self.transformer = CausalTransformerDecoder(decoder_layer, num_layers)
    self.output = nn.Linear(dim_model, vocab_size - 1)  # excluding the <bos> token

  def forward(self, x, kv_cache=None):
    N, S = x.shape
    # (N, S) -> (N, S, D)
    x = self.embedding(x)
    x = x + self.positional_encoding[:S]
    mask = torch.tril(torch.ones((S, S), device=x.device))
    x = self.transformer(x, mask=mask, kv_cache=kv_cache)
    # (N, S, D) -> (N, S)
    scores = self.output(x)
    return scores

  def loss(self, x):
    N, S = x.shape
    target = x.long()
    # prepend <bos> token at the beginning
    # use self.vocab_size - 1, not -1; Embedding needs positive indices
    bos_token = torch.tensor(self.vocab_size - 1, device=x.device)
    x = torch.cat((bos_token.expand(N, 1), x[:, :-1]), dim=1)
    
    scores = self.forward(x)
    return F.cross_entropy(scores.permute(0, 2, 1), target)
  
  def samples(self, num_samples):
    x = torch.zeros((num_samples, self.max_length), dtype=torch.long, device=self.positional_encoding.device)
    x[:, 0] = self.vocab_size - 1
    with torch.no_grad():
      for i in tqdm(range(self.max_length), desc="Generating samples"):
        scores = self.forward(x[:, :i+1])  # (N, i+1, vocab_size-1)
        next_token_scores = scores[:, -1]  # (N, vocab_size-1)
        probs = torch.softmax(next_token_scores, dim=1)
        next_token = torch.multinomial(probs, num_samples=1)  # (N, 1)
        if i < self.max_length - 1:
          x[:, i+1] = next_token.view(-1)
        else:
          samples = torch.cat((x[:, 1:], next_token), dim=1)
    return samples


class CausalTransformerWithKVCache(CausalTransformer):
  def samples_with_timing(self, num_samples, use_kv_cache=True):
    x = torch.zeros((num_samples, self.max_length), dtype=torch.long, device=self.positional_encoding.device)
    x[:, 0] = self.vocab_size - 1
    kv_cache = {} if use_kv_cache else None

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.max_length)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.max_length)]
    time_list = []

    with torch.no_grad():
      for i in tqdm(range(self.max_length), desc="Generating samples"):
        start_event[i].record()

        scores = self.forward(x[:, :i+1], kv_cache=kv_cache)  # use kv_cache
        next_token_scores = scores[:, -1]
        probs = torch.softmax(next_token_scores, dim=1)
        next_token = torch.multinomial(probs, num_samples=1)
        if i < self.max_length - 1:
          x[:, i+1] = next_token.view(-1)
        else:
          samples = torch.cat((x[:, 1:], next_token), dim=1)
        
        end_event[i].record()
        torch.cuda.synchronize()  # both events must be completed before calculating elapsed time.
        time_list.append(start_event[i].elapsed_time(end_event[i]))
    return time_list, samples

  def samples(self, num_samples, use_kv_cache=True):
    x = torch.zeros((num_samples, self.max_length), dtype=torch.long, device=self.positional_encoding.device)
    x[:, 0] = self.vocab_size - 1
    kv_cache = {} if use_kv_cache else None
    with torch.no_grad():
      for i in tqdm(range(self.max_length), desc="Generating samples"):
        scores = self.forward(x[:, :i+1], kv_cache=kv_cache)  # use kv_cache
        next_token_scores = scores[:, -1]
        probs = torch.softmax(next_token_scores, dim=1)
        next_token = torch.multinomial(probs, num_samples=1)
        if i < self.max_length - 1:
          x[:, i+1] = next_token.view(-1)
        else:
          samples = torch.cat((x[:, 1:], next_token), dim=1)
    return samples