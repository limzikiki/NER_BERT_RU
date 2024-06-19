"""
Made for Subject in University of Latvia Natural Language Processing Basics

By:
Leonards Tesnovs LT21026
Sofya Yulpatova SY21002

"""

# pip install transformers seqeval keras torch matplotlib seaborn tensorflow Keras-Preprocessing
# Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process
# .\venv\Scripts\activate


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, classification_report

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import seaborn as sns


# graph
plt.rcParams["figure.figsize"] = (12,6)




device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

print(device.type)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))




sentences, labels = [], []
tag_values = set() # set of the values found 
with open('aij-wikiner-ru-wp3', encoding="utf8") as df:
    for line in df:
        sent_w = [] # for word
        # Tags correspons to this https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
        sent_t = [] # for tag
        sent = line.split()
        for s in sent:
            tag = s.split('|')
            # getting rid of the word types, collecting only word and tag, putting them into separate groups
            sent_w.append(tag[0])  
            sent_t.append(tag[2]) 
            if len(tag[2]) > 0:
                tag_values.update([tag[2]])
        
        # Basically this if statment is to check when we have empty line
        if len(sent_w) > 0:
            sentences.append(sent_w)
            labels.append(sent_t)
            

assert len(sentences) == len(labels)

# print(sentences[0])just for validity check

# Add padding to the tag values, to ensure that all the imputs fed to the model are the same size
tag_values = list(tag_values)
tag_values.append('PAD')
tag_values.append('X')
tag2idx = {t: i for i, t in enumerate(tag_values)}
#tag2idx["X"] = -100.0
# print(tag2idx) just for check
"""
subword-based tokenization which is a solution between word and character-based tokenization. 
The main idea is to solve the issues faced by word-based tokenization (very large vocabulary size, 
large number of OOV tokens, and different meaning of very similar words) and character-based tokenization 
(very long sequences and less meaningful individual tokens).
The subword-based tokenization algorithms uses the following principles.
1 Do not split the frequently used words into smaller subwords.
2 Split the rare words into smaller meaningful subwords.
"""

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


"""
We put value X after the beginning of the token. Because BERT considers every input impacts the loss function. 
So we set the X to -100 and later on we will ignore padding as well.

Question for further: 
1. Should we put both in mask or as -100? 
2. how those approaches impact precision? 
"""
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Use the BERT tokenizer to tokenize the word
        tokenized_word = tokenizer.tokenize(word)
        
        # Extend the tokenized sentence with the tokenized word
        tokenized_sentence.extend(tokenized_word)
                                                                                                                                                                                         
        # Assign the label to the first token and 'X' to the subsequent sub-tokens
        labels.extend([label] + ['X'] * (len(tokenized_word) - 1))
    
    return tokenized_sentence, labels


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]


lenths = [len(sent) for sent, _ in tokenized_texts_and_labels]
print(min(lenths), np.mean(lenths), max(lenths), sep='|')
print(plt.hist(lenths))
# plt.show()

import time
plt.savefig(f"books_read{time.time_ns()}.png")

# max size of sequence set to 
MAX_LEN = 200
# batch size, bert advice 32
BATCH_SIZE = 32


tokenized_texts = [token for token, _ in tokenized_texts_and_labels]
labels = [label for _, label in tokenized_texts_and_labels]

assert len(tokenized_texts) == len(labels)



print(tokenized_texts[0])
print(labels[0])

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN,
                          dtype='long',
                          value=0.0,
                          truncating='post',
                          padding='post')
     

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN,
                     value=tag2idx['PAD'],
                     padding='post',
                     dtype='long',
                     truncating='post')


# Create mask for ignoring the paddings
attention_masks = [[float(i != 0.0 and i != tag2idx['X']) for i in ii] for ii in input_ids]



tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=42, test_size=0.02)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=42, test_size=0.02)

# To Tensors
# Tensors are optimized for high-performance computing on GPUs. 
# By converting data to tensors, we can leverage GPU acceleration to significantly speed up the 
# training and inference processes.

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags, dtype=torch.long)
val_tags = torch.tensor(val_tags, dtype=torch.long)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
     

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags) #Combines multiple tensors into a single dataset object
train_sampler = RandomSampler(train_data) # Define the strategy for sampling elements from the dataset during training and validation.
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data) # Define the strategy for sampling elements from the dataset during training and validation.
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)
# DataLoader  -- takes a dataset and a sampler and handles batching, shuffling, and parallel data loading. 
# It simplifies the process of feeding data to the model during training and evaluation


"""RandomSampler shuffles the data at the beginning of each epoch, ensuring that the model does not overfit to the order of the training data. This randomness helps improve the generalization of the model.
SequentialSampler iterates over the dataset in a sequential order, which is typically used for validation or test sets where we want to maintain the order for consistency."""






# lOAD MODEL

model = BertForTokenClassification.from_pretrained(
    'bert-base-cased',
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)


model.to(device)



epochs = 4
max_grad_norm = 1.0


# TO change the learning rate 
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

 

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
loss_values, validation_loss_values = [], []



# Training it self
for _ in trange(epochs, desc='Epoch'):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()

        # clip_grad_norm_ помогает против взрыва градиентов
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print("Средний лосс обучения: {}".format(avg_train_loss))

    loss_values.append(avg_train_loss)

    # Валидация
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        eval_loss += outputs[0].mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    validation_loss_values.append(eval_loss)
    print("Лосс на валидации: {}".format(eval_loss))
    print("Accuracy на валидации: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = []
    #pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
    #       for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]

    for p, l in zip(predictions, true_labels):
        temp_tags = []
        for p_i, l_i in zip(p, l):
            if tag_values[l_i] == "PAD" or tag_values == "X":
                continue
            temp_tags.append(tag_values[p_i])
        pred_tags.append(temp_tags)
         
    valid_tags = []
    for l in true_labels:
        temp_tags = []
        for l_i in l:
            if tag_values[l_i] == "PAD" or tag_values == "X":
                continue
            temp_tags.append(tag_values[l_i])
        valid_tags.append(temp_tags)

    # valid_tags = [tag_values[l_i] for l in true_labels
    #                               for l_i in l if tag_values[l_i] != "PAD"]
    
    try:
        print("F1 на валидации: {}".format(f1_score(valid_tags, pred_tags)))
    except Exception:
        print("Failed to compute f1")
        print(valid_tags, pred_tags)
    print()

print(classification_report(valid_tags, pred_tags))



plt.plot(loss_values, 'b-o', label='training loss')
plt.plot(validation_loss_values, 'r-o', label='validation loss')

plt.title('Learning curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


test_sentence = """
Судно-трубоукладчик «Академик Черский», которое рассматривали как один 
из вариантов завершения строительства газопровода «Северный поток — 2», 
указало курс на Находку. Об этом информирует 
РИА Новости со ссылкой на данные порталов по отслеживанию судов 
Marine Traffic и Vesselfinder.
"""

tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()



with torch.no_grad():
    output = model(input_ids)
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

# объединяем токены и метки
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)




for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))
    
model.save_pretrained("./model/")
tokenizer.save_pretrained("./model/")