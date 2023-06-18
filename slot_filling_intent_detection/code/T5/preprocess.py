import pandas as pd
import os
import pickle

import argparse

def create_data(path, mode):
    file = open(path)
    lines = file.readlines()
    input_text, target_text = [], []
    sample_lines, unique_texts = {}, set()
    # m=0
    redandent_line = False
    for line in lines:
        if line == '\n':
            # m +=1
            # if m>1000:
            #     break
            if sample_lines:
                if mode['slot_filling'] and mode['intent_detection']:
                    input_text.append("intent and slot: " + sample_lines['text'])
                    target_sample = sample_lines['intent'] + "; " + " ".join(sample_lines['slots'])
                elif mode['slot_filling']:
                    input_text.append("slot: " + sample_lines['text'])
                    target_sample = " ".join(sample_lines['slots'])
                elif mode['intent_detection']:
                    input_text.append("intent: " + sample_lines['text'])
                    target_sample = sample_lines['intent']
                else:
                    print("one of slot_filling and intent_detection or both must be True")
                    break
                target_text.append(target_sample)
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
            elif 'slots' not in line:
                if 'slots' not in sample_lines.keys():
                    sample_lines['slots'] = []
                slot = line.split('\t')[3].strip()
                sample_lines['slots'].append(slot)

    data = pd.DataFrame(list(zip(input_text, target_text)), columns =['source_text', 'target_text'])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intent_detection', default=True, required=True)
    parser.add_argument('--slot_filling', default=True, required=True)
    args = parser.parse_args()
    mode = {'intent_detection':args.intent_detection=='True', 'slot_filling':args.slot_filling=='True'}

    print("reading train and val data..")
    train_data = create_data(os.path.join('dataset', 'train-en.conllu'), mode)
    val_data = create_data(os.path.join('dataset', 'development-en.conllu'), mode)
    print("reading test data..")
    test_data = create_data(os.path.join('dataset', 'test-en.conllu'), mode)

    print("number of data after remove redundant texts:")
    print(f"\t train data: {len(train_data['source_text'])}")
    print(f"\t val data: {len(val_data['source_text'])}")
    print(f"\t test data: {len(test_data['source_text'])}")

    print("save prepared data..")
    train_data.to_csv(os.path.join('T5', 'data', 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join('T5', 'data', 'val_data.csv'), index=False)
    test_data.to_csv(os.path.join('T5', 'data', 'test_data.csv'), index=False)

    # with open(os.path.join('T5', 'data', 'train_data.pth'), "wb") as outfile:
    #     pickle.dump(train_data, outfile)
    # with open(os.path.join('T5', 'data', 'val_data.pth'), "wb") as outfile:
    #     pickle.dump(val_data, outfile)
    # with open(os.path.join('T5', 'data', 'test_data.pth'), "wb") as outfile:
    #     pickle.dump(test_data, outfile)

    print("data is saved in T5/data/")