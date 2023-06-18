import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import os
import pickle

def create_data(path):
    file = open(path)
    lines = file.readlines()
    input_data, intents, slots = [], [], []
    intent_set, slot_set, unique_texts = set(), set(), set()
    sample_lines = {}
    # m=0
    redandent_line = False
    for line in lines:
        if line == '\n':
            # m +=1
            # if m>1000:
            #     break
            if sample_lines:
                input_data.append(sample_lines['text'])
                intents.append(sample_lines['intent'])
                slots.append(sample_lines['slots'])
            sample_lines = {}
            redandent_line = False
        elif redandent_line:
            continue
        else:
            if 'text:' in line:
                text = line[line.find(':')+1:].strip()
                text = text.lower()
                if text in unique_texts:
                    redandent_line = True
                    continue
                sample_lines['text'] = text
                unique_texts.add(text)
            elif 'intent' in line:
                sample_lines['intent'] = line.split()[-1].strip()
                intent_set.add(sample_lines['intent'])
            elif 'slots' not in line:
                if 'slots' not in sample_lines.keys():
                    sample_lines['slots'] = []
                slot = line.split('\t')[3].strip()
                sample_lines['slots'].append(slot)
                slot_set.add(slot)
    max_input_len = len(max(input_data, key=lambda x:len(x.split())).split())+7  ## 7 is for spliting some tokens in tokenizer
    intent2id = {intent:id for id, intent in enumerate(['pad', 'unknown'] + sorted(list(intent_set)))}
    id2intent = {id:intent for intent, id in intent2id.items()}
    slot2id = {slot:id for id, slot in enumerate(['pad', 'unknown'] + sorted(list(slot_set)))}
    id2slot = {id:slot for slot, id in slot2id.items()}
    info = {'intent2id': intent2id, 'id2intent': id2intent, 'slot2id': slot2id, 'id2slot': id2slot, 'max_input_len':max_input_len}
    return input_data, intents, slots, info

def create_tensor_data(tokenizer, input_data, intents, slots, info):
    encoded_dict = tokenizer(
                    input_data,  # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length=info['max_input_len'],
                    padding = 'max_length',
                    # truncation=True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
              )
    input_tensor = encoded_dict['input_ids']
    attention_masks_tensor = encoded_dict['attention_mask']
    token_type_tensor = encoded_dict['token_type_ids']
    slot_ids = []
    intent_ids = []
    for i, text in enumerate(input_data):
        words = text.split()
        slot_ids_line = []
        for j, word in enumerate(words):
            word_tokenized = tokenizer(word, add_special_tokens = False)
            # print(word_tokenized)
            for k in range(len(word_tokenized['input_ids'])):
                if k > 0 and 'B-' in slots[j]:
                    slot = slots[i][j].replace('B-', 'I-')
                    if slot in info['slot2id'].keys():
                        slot_ids_line.append(info['slot2id'][slot])
                    else:
                        slot_ids_line.append(info['slot2id']['unknown'])

                else:
                    if slots[i][j] in info['slot2id'].keys():
                        slot_ids_line.append(info['slot2id'][slots[i][j]])
                    else:
                        slot_ids_line.append(info['slot2id']['unknown'])
        if len(slot_ids_line) < info['max_input_len']-2:
            for _ in range(info['max_input_len'] - len(slot_ids_line) - 2):
                slot_ids_line.append(0)
        elif len(slot_ids_line) > info['max_input_len']-2:
            slot_ids_line = slot_ids_line[:info['max_input_len']-2]
        if intents[i] in info['intent2id']:
            intent_ids.append(info['intent2id'][intents[i]])
        else:
            intent_ids.append(info['intent2id']['unknown'])
        slot_ids.append(torch.Tensor(slot_ids_line).long())

    # Convert the lists into tensors.
    slots_tensor = pad_sequence(slot_ids, batch_first=True)
    intent_tensor = torch.Tensor(intent_ids).long()
    return input_tensor, slots_tensor, intent_tensor, attention_masks_tensor, token_type_tensor

def create_dataloader(input_tensor, slots_tensor, intent_tensor, attention_masks_tensor, token_type_tensor, batch_size):
    dataset = TensorDataset(input_tensor, attention_masks_tensor, token_type_tensor, slots_tensor, intent_tensor)
    # We'll take training samples in random order. 
    dataloader = DataLoader(
                dataset,  # The training samples.
                sampler = RandomSampler(dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    return dataloader

if __name__ == "__main__":
    print('load bert tokenizer..')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("reading train and val data..")
    train_input_data, train_intents, train_slots, info = create_data(os.path.join('dataset', 'train-en.conllu'))
    val_input_data, val_intents, val_slots, _ = create_data(os.path.join('dataset', 'development-en.conllu'))
    print("reading test data..")
    test_input_data, test_intents, test_slots, _ = create_data(os.path.join('dataset', 'test-en.conllu'))

    # m, n = 0, 0
    # for t in val_input_data:
    #     if t not in train_input_data:
    #         m += 1
    # for t in test_input_data:
    #     if t not in train_input_data:
    #         n += 1
    #         if n < 10:
    #             print(t)
    # print(f"m={m},  n={n}")

    print("create train and val tensor data..")
    train_input_tensor, train_slots_tensor, train_intent_tensor, train_attention_masks_tensor, train_token_type_tensor = create_tensor_data(tokenizer, train_input_data, train_intents, train_slots, info)
    val_input_tensor, val_slots_tensor, val_intent_tensor, val_attention_masks_tensor, val_token_type_tensor = create_tensor_data(tokenizer, val_input_data, val_intents, val_slots, info)
    print("create test tensor data..")
    test_input_tensor, test_slots_tensor, test_intent_tensor, test_attention_masks_tensor, test_token_type_tensor = create_tensor_data(tokenizer, test_input_data, test_intents, test_slots, info)

    print("data shape after remove redundant texts:")
    print(f"\t train data: {train_input_tensor.shape}")
    print(f"\t val data: {val_input_tensor.shape}")
    print(f"\t test data: {test_input_tensor.shape}")

    print("create dataloader for train and val..")
    train_dataloader = create_dataloader(train_input_tensor, train_slots_tensor, train_intent_tensor, train_attention_masks_tensor, train_token_type_tensor, 32)
    val_dataloader = create_dataloader(val_input_tensor, val_slots_tensor, val_intent_tensor, val_attention_masks_tensor, val_token_type_tensor, 32)
    print("create dataloader for test..")
    test_dataloader = create_dataloader(test_input_tensor, test_slots_tensor, test_intent_tensor, test_attention_masks_tensor, test_token_type_tensor, 32)

    print("save prepared data..")
    with open(os.path.join('bert', 'data', 'train_dataloader.pth'), "wb") as outfile:
        pickle.dump(train_dataloader, outfile)
    with open(os.path.join('bert', 'data', 'val_dataloader.pth'), "wb") as outfile:
        pickle.dump(val_dataloader, outfile)
    with open(os.path.join('bert', 'data', 'test_dataloader.pth'), "wb") as outfile:
        pickle.dump(test_dataloader, outfile)
    with open(os.path.join('bert', 'data', "info.p"), "wb") as outfile:
        pickle.dump(info, outfile)
    print("data is saved in bert/data/")

