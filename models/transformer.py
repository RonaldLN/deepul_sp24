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
      # compute for uncached prompt
      K_T = self.key(key).view(N, T, H, D_H).permute(0, 2, 3, 1)
      V = self.value(value).view(N, T, H, D_H).permute(0, 2, 1, 3)
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
    for i, mod in enumerate(self.layers):
      layer_cache = kv_cache[i] if kv_cache is not None else None
      output = mod(output, mask=mask, kv_cache=layer_cache)
    return output


class CausalTransformer(nn.Module):
  def __init__(self, vocab_size, dim_model, num_heads, num_layers,
               dim_feedforward, max_length, dropout=0.1):
    super().__init__()
    self.vocab_size = vocab_size
    # use self.vocab_size - 1, not -1; Embedding needs positive indices
    self._bos = vocab_size - 1
    self._null = 0  # use 0 as <null> for image
    self.max_length = max_length
    self.num_layers = num_layers

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
    bos_token = self._bos * torch.ones((N, 1), dtype=torch.long, device=x.device)
    x = torch.cat((bos_token, x[:, :-1]), dim=1)
    
    scores = self.forward(x)
    return F.cross_entropy(scores.permute(0, 2, 1), target)
  
  def samples(self, num_samples):
    x = self._null * torch.ones((num_samples, self.max_length), dtype=torch.long, device=self.positional_encoding.device)
    x[:, 0] = self._bos
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
    x = self._null * torch.ones((num_samples, self.max_length), dtype=torch.long, device=self.positional_encoding.device)
    x[:, 0] = self._bos
    kv_cache = [{} for _ in range(self.num_layers)] if use_kv_cache else None

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
    x = self._null * torch.ones((num_samples, self.max_length), dtype=torch.long, device=self.positional_encoding.device)
    x[:, 0] = self._bos
    kv_cache = [{} for _ in range(self.num_layers)] if use_kv_cache else None
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


class TextCausalTransformer(CausalTransformerWithKVCache):
  def __init__(self, char_to_idx, dim_model, num_heads, num_layers,
               dim_feedforward, max_length, dropout=0.1):
    vocab_size = len(char_to_idx)
    super().__init__(vocab_size, dim_model, num_heads, num_layers,
                     dim_feedforward, max_length, dropout)
    # override <bos> and <null> tokens for text data
    self._bos = char_to_idx['<bos>']
    self._null = char_to_idx['<null>']
  
  def loss(self, x, target):
    scores = self.forward(x)
    return F.cross_entropy(scores.permute(0, 2, 1), target.long())


class MultiModalCausalTransformer(CausalTransformerWithKVCache):
  def __init__(self, word_to_idx, img_n_embeddings, dim_model, num_heads, num_layers,
               dim_feedforward, words_length, img_length, dropout=0.1):
    vocab_size = len(word_to_idx) + img_n_embeddings + 3  # +3 for <bos>, <end of text> and <end of image> tokens
    max_length = words_length + img_length + 2  # +2 for <end of text> and <end of image> tokens
    super().__init__(vocab_size, dim_model, num_heads, num_layers,
                     dim_feedforward, max_length, dropout)

    self.word_vocab_size = len(word_to_idx)
    self.words_length = words_length
    self.img_length = img_length
    # <bos> token is already set to vocab_size - 1
    self._end_of_text = vocab_size - 3
    self._end_of_image = vocab_size - 2

    # masks for inference
    out_vocab_size = vocab_size - 1  # excluding <bos> token
    self.register_buffer('text_mask', torch.ones(out_vocab_size, dtype=torch.long))
    self.text_mask[self.word_vocab_size:] = 0
    self.register_buffer('image_mask', torch.zeros(out_vocab_size, dtype=torch.long))
    self.image_mask[self.word_vocab_size:self.word_vocab_size + img_n_embeddings] = 1
    self.register_buffer('ends_mask', torch.zeros(out_vocab_size, dtype=torch.long))
    self.ends_mask[[self._end_of_text, self._end_of_image]] = 1

  def loss(self, img_x, text_x):
    N = img_x.shape[0]
    end_of_text_token = self._end_of_text * torch.ones((N, 1), dtype=torch.long, device=img_x.device)
    end_of_image_token = self._end_of_image * torch.ones((N, 1), dtype=torch.long, device=img_x.device)

    img_x = img_x + self.word_vocab_size  # shift image token indices

    if torch.rand(1).item() < 0.5:
      # text -> image
      x = torch.cat((end_of_image_token, text_x, end_of_text_token, img_x), dim=1)
    else:
      # image -> text
      x = torch.cat((end_of_text_token, img_x, end_of_image_token, text_x), dim=1)
    
    return super().loss(x)

  def samples(self, num_samples, use_kv_cache=True, image_prompt=None, text_prompt=None):
    x = self._null * torch.ones((num_samples, self.max_length), dtype=torch.long, device=self.positional_encoding.device)
    x[:, 0] = self._bos
    kv_cache = [{} for _ in range(self.num_layers)] if use_kv_cache else None
    if image_prompt is not None:
      # shift image prompt
      image_prompt = image_prompt + self.word_vocab_size

      x[:, 1] = self._end_of_text
      x[:, 2:2+self.img_length] = image_prompt
      x[:, 2+self.img_length] = self._end_of_image

      start_idx = self.img_length + 2  # <bos> <end of text> image <end of image>
      samples_image, samples_text = self._samples_image_first(x, start_idx, kv_cache)
    elif text_prompt is not None:
      x[:, 1] = self._end_of_image
      x[:, 2:2+self.words_length] = text_prompt
      x[:, 2+self.words_length] = self._end_of_text

      start_idx = self.words_length + 2  # <bos> <end of image> text <end of text>
      samples_image, samples_text = self._samples_text_first(x, start_idx, kv_cache)
    else:  # no prompt
      with torch.no_grad():
        # sample <end of text> or <end of image>
        scores = self.forward(x[:, :1], kv_cache=kv_cache)
        next_token_scores = scores[:, -1]

        # only <end of text> or <end of image>
        next_token_scores = next_token_scores.masked_fill(self.ends_mask == 0, -1e10)

        probs = torch.softmax(next_token_scores, dim=1)
        next_token = torch.multinomial(probs, num_samples=1)
        x[:, 1] = next_token.view(-1)

      # determine which modality to sample first based on the sampled token
      group_mask = (x[:, 1] == self._end_of_text)  # (N,)
      text_first_x = x[group_mask]  # (N_1, max_length)
      image_first_x = x[~group_mask]  # (N_2, max_length)

      start_idx = 1  # <bos> <end of text/image>
      kv_cache1 = [{} for _ in range(self.num_layers)] if use_kv_cache else None
      kv_cache2 = [{} for _ in range(self.num_layers)] if use_kv_cache else None
      samples_image1, samples_text1 = self._samples_text_first(text_first_x, start_idx, kv_cache1)
      samples_image2, samples_text2 = self._samples_image_first(image_first_x, start_idx, kv_cache2)

      # merge samples
      samples_image = torch.cat((samples_image1, samples_image2), dim=0)
      samples_text = torch.cat((samples_text1, samples_text2), dim=0)

    samples_image = samples_image - self.word_vocab_size  # shift back image token indices
    return samples_image, samples_text

  def _samples_text_first(self, x, start_idx, kv_cache):
    milestone = 2 + self.words_length
    with torch.no_grad():
      for i in tqdm(range(start_idx, self.max_length), desc="Generating samples"):
        scores = self.forward(x[:, :i+1], kv_cache=kv_cache)
        next_token_scores = scores[:, -1]

        # index i+1 is the target (to be generated)
        if i+1 < milestone:
          next_token_scores = next_token_scores.masked_fill(self.text_mask == 0, -1e10)
        elif i+1 == milestone:
          next_token_scores = next_token_scores.masked_fill(self.ends_mask == 0, -1e10)
        else:
          next_token_scores = next_token_scores.masked_fill(self.image_mask == 0, -1e10)

        probs = torch.softmax(next_token_scores, dim=1)
        next_token = torch.multinomial(probs, num_samples=1)
        if i < self.max_length - 1:
          x[:, i+1] = next_token.view(-1)
        else:
          samples = torch.cat((x[:, 2:], next_token), dim=1)
      # split text and image
      samples_text, _, samples_image = torch.split(samples, (self.words_length, 1, self.img_length), dim=1)
    return samples_image, samples_text
  
  def _samples_image_first(self, x, start_idx, kv_cache):
    milestone = 2 + self.img_length
    with torch.no_grad():
      for i in tqdm(range(start_idx, self.max_length), desc="Generating samples"):
        scores = self.forward(x[:, :i+1], kv_cache=kv_cache)
        next_token_scores = scores[:, -1]

        # index i+1 is the target (to be generated)
        if i+1 < milestone:
          next_token_scores = next_token_scores.masked_fill(self.image_mask == 0, -1e10)
        elif i+1 == milestone:
          next_token_scores = next_token_scores.masked_fill(self.ends_mask == 0, -1e10)
        else:
          next_token_scores = next_token_scores.masked_fill(self.text_mask == 0, -1e10)

        probs = torch.softmax(next_token_scores, dim=1)
        next_token = torch.multinomial(probs, num_samples=1)
        if i < self.max_length - 1:
          x[:, i+1] = next_token.view(-1)
        else:
          samples = torch.cat((x[:, 2:], next_token), dim=1)
      # split image and text
      samples_image, _, samples_text = torch.split(samples, (self.img_length, 1, self.words_length), dim=1)
    return samples_image, samples_text