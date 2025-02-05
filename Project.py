import torch
import numpy as np
import time
import math
from torch import nn
import torch.nn.functional as F
START = '<START>'
PADDING = '<PAD>'
END = '<END>'

# Hindi vocabulary with special tokens
hindi_vocabulary = [START, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                    '०', '१', '२', '३', '४', '५', '६', '७', '८', '९', 
                    ':', '<', '=', '>', '?', '@', 
                    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 
                    'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 
                    'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 
                    'ल', 'व', 'श', 'ष', 'स', 'ह', 'ळ', 'क्ष', 'ज्ञ', 
                    '[', '\\', ']', '^', '_', '`', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', '्', 'ं', 'ः', 
                    '{', '|', '}', '~', PADDING, END]

# Gujarati vocabulary with special tokens
gujarati_vocabulary = [START, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                       '૦', '૧', '૨', '૩', '૪', '૫', '૬', '૭', '૮', '૯', 
                       ':', '<', '=', '>', '?', '@', 
                       'અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'એ', 'ઐ', 'ઓ', 'ઔ', 'ક', 'ખ', 
                       'ગ', 'ઘ', 'ચ', 'છ', 'જ', 'ઝ', 'ઞ', 'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 
                       'ત', 'થ', 'દ', 'ધ', 'ન', 'પ', 'ફ', 'બ', 'ભ', 'મ', 'ય', 'ર', 
                       'લ', 'વ', 'શ', 'ષ', 'સ', 'હ', 'ળ', 'ક્ષ', 'જ્ઞ', 
                       '[', '\\', ']', '^', '_', '`', 'ા', 'િ', 'ી', 'ુ', 'ૂ', 'ે', 'ૈ', 'ો', 'ૌ', '્', 'ં', 'ઃ', 
                       '{', '|', '}', '~', PADDING, END]

# Sample text for processing
text = 'ઉઝઝસી'

list(text)
index_to_guj = {k:v for k,v in enumerate(gujarati_vocabulary)}
guj_to_index = {v:k for k,v in enumerate(gujarati_vocabulary)}
index_to_hi = {k:v for k,v in enumerate(hindi_vocabulary)}
hi_to_index = {v:k for k,v in enumerate(hindi_vocabulary)}
hindi_file = 'train.hi'
gujarati_file = 'train.gu'
with open(hindi_file, 'r', encoding='utf-8') as file:
    hindi_sentences = file.readlines()
    
with open(gujarati_file, 'r', encoding='utf-8') as file:
    gujarati_sentences = file.readlines()

total_sentences = 768038
hindi_sentences = hindi_sentences[:total_sentences]
gujarati_sentences = gujarati_sentences[:total_sentences]

hindi_sentences = [sentence.rstrip('\n') for sentence in hindi_sentences]
gujarati_sentences = [sentence.rstrip('\n') for sentence in gujarati_sentences]
hindi_sentences[0:10]
gujarati_sentences[0:10]
max(max(len(x) for x in hindi_sentences), max(len(x) for x in gujarati_sentences))
PERCENTILE = 97
print(np.percentile([len(x) for x in hindi_sentences], PERCENTILE))
print(np.percentile([len(x) for x in gujarati_sentences], PERCENTILE))
max_seq_length = 200

def is_valid_tokens(sentence,vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence,max_seq_length):
    return len(list(sentence)) < (max_seq_length-1)

valid_sentence_indices = []

for index in range(len(gujarati_sentences)):
    gujarati_sentence, hindi_sentence = gujarati_sentences[index], hindi_sentences[index]
    if is_valid_length(gujarati_sentence,max_seq_length) \
    and is_valid_length(hindi_sentence,max_seq_length) \
    and is_valid_tokens(gujarati_sentence,gujarati_vocabulary) \
    and is_valid_tokens(hindi_sentence,hindi_vocabulary):
        valid_sentence_indices.append(index)
    
print(f"Number of total gujarati sentences: {len(gujarati_sentences)}")
print(f"Number of valid gujarati sentences: {len(valid_sentence_indices)}")
gujarati_sentences = [gujarati_sentences[i] for i in valid_sentence_indices]
hindi_sentences = [hindi_sentences[i] for i in valid_sentence_indices]
gujarati_sentences[0:5]
hindi_sentences[0:5]
# Dataset creation
from torch.utils.data import Dataset, DataLoader
class TextDataSet(Dataset):
    def __init__(self, hindi_sentences, gujarati_sentences):
        super().__init__
        self.hindi_sentences = hindi_sentences
        self.gujarati_sentences = gujarati_sentences
        
    def __len__(self):
        return len(self.hindi_sentences)
    
    def __getitem__(self,idx):
        return self.hindi_sentences[idx], self.gujarati_sentences[idx]
dataset = TextDataSet(hindi_sentences,gujarati_sentences)    
len(dataset)
dataset[22]
batch_size=32
train_loader = DataLoader(dataset,batch_size)
iterator = iter(train_loader)

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(max_sequence_length, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size() # in practice, this is the same for both languages...so we can technically combine with normal attention
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask) # We don't need the mask for cross attention, removing in outer function!
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                kn_vocab_size,
                hindi_to_index,
                kannada_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, hindi_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
gj_vocab_size = len(gujarati_vocabulary)

transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          gj_vocab_size,
                          hi_to_index,
                          guj_to_index,
                          START, 
                          END, 
                          PADDING)
len(guj_to_index)
NEG_INFTY = -1e9

def create_masks(hi_batch, gj_batch):
    num_sentences = len(hi_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        hi_sentence_length, gj_sentence_length = len(hi_batch[idx]), len(gj_batch[idx])
        hi_chars_to_padding_mask = np.arange(hi_sentence_length + 1, max_sequence_length)
        gj_chars_to_padding_mask = np.arange(gj_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, hi_chars_to_padding_mask] = True
        encoder_padding_mask[idx, hi_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, gj_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, gj_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, hi_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, gj_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


# Initialize model, optimizer, and loss function
criterian = nn.CrossEntropyLoss(ignore_index=guj_to_index[PADDING], reduction='none')
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transformer.train()
transformer.to(device)
# Training loop

def save_model(epoch, model, optim, loss, model_save_path, optim_save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
    }, model_save_path)
    print(f"Model saved at {model_save_path}")

num_epochs = 25
model_save_path = "transformer_model.pth"
optim_save_path = "transformer_optimizer.pth"

total_training_start = time.time()

for epoch in range(num_epochs):
    try:
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()
        transformer.train()
        total_loss = 0

        for batch_num, batch in enumerate(train_loader):
            hi_batch, gj_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(hi_batch, gj_batch)

            optim.zero_grad()
            gj_predictions = transformer(
                hi_batch,
                gj_batch,
                encoder_self_attention_mask.to(device),
                decoder_self_attention_mask.to(device),
                decoder_cross_attention_mask.to(device),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True,
            )
            labels = transformer.decoder.sentence_embedding.batch_tokenize(gj_batch, start_token=False, end_token=True)
            loss = criterian(
                gj_predictions.view(-1, gj_vocab_size).to(device),
                labels.view(-1).to(device),
            )
            valid_indices = torch.where(labels.view(-1) == guj_to_index[PADDING], False, True)
            loss = loss.sum() / valid_indices.sum()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if batch_num % 100 == 0:
                print(f"Iteration {batch_num}, Loss: {loss.item()}")

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader)}")
        print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

        # Save model and optimizer state after each epoch
        save_model(epoch, transformer, optim, total_loss / len(train_loader), model_save_path, optim_save_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        save_model(epoch, transformer, optim, total_loss / len(train_loader), model_save_path, optim_save_path)
        break

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")

# Translation function
transformer.eval()

def translate(hi_sentence):
    hi_sentence = (hi_sentence,)
    gj_sentence = ("",)
    for word_counter in range(max_sequence_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(hi_sentence, gj_sentence)
        predictions = transformer(
            hi_sentence,
            gj_sentence,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=False,
        )
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_guj[next_token_index]
        gj_sentence = (gj_sentence[0] + next_token,)
        if next_token == END:
            break
    return gj_sentence[0]

translation = translate("एक टीम ने एक साथ काम किया")
print(f"Translation: {translation}")

# Loading the model (if needed)
checkpoint = torch.load(model_save_path)
transformer.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
transformer.to(device)
print(f"Model loaded from epoch {epoch} with loss {loss:.4f}")