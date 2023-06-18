import os
import pickle
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from simplet5 import SimpleT5
from result.eval import conlleval

def show_scores(words, mode, predicted_slots=None, predicted_intents=None, real_slots=None, real_intents=None):
    print(mode)
    if mode['slot_filling']:
        slot_scores = conlleval(predicted_slots, real_slots, words, 'out.txt')
        print(f"slot scores: [precision:{slot_scores['p']}, recall:{slot_scores['r']}, F1:{slot_scores['f1']}]")
    if mode['intent_detection']:
        intent_accuracy = accuracy_score(real_intents, predicted_intents)
        print("intent accuracy: {0:.3f}".format(intent_accuracy))

def predict(model, text_input):
    result_text = model.predict(text_input)
    return result_text

def write_results(path, input_text, target_text, predicted_text):
    f = open(path, 'w')
    for i in range(len(input_text)):
        f.write('input text: ' + str(input_text[i]) + '\n')
        f.write('target text: ' + str(target_text[i]) + '\n')
        f.write('predicted text: ' + str(predicted_text[i]) + '\n')
        f.write('#########################\n')
    f.close()

if __name__ == "__main__":
    print("load test data..")
    test_data = pd.read_csv(os.path.join('T5', 'data', 'test_data.csv'))

    print("load model..")
    model_path = os.path.join("T5", "model", "simplet5-epoch-2-*")
    model_path = glob.glob(model_path)[0]
    print(f'model path: {model_path}')
    model = SimpleT5()
    # model.from_pretrained(model_type="t5", model_name="t5-base")
    model.load_model("t5", model_path, use_gpu=True)
    predicted_text = []
    predicted_slots = []
    predicted_intents = []
    real_slots = []
    real_intents = []
    real_words = []
    mode = {'intent_detection': 'intent' in test_data['source_text'][0], 'slot_filling': 'slot' in test_data['source_text'][0]}
    print("evaluate test data..")

    test_data = test_data.head(1000)
    for index, row in tqdm(test_data.iterrows(), leave=True):
        result_text = predict(model, row['source_text'])[0]
        # print( row['source_text'])
        # print(row['target_text'])
        # print(result_text)
        predicted_text.append(result_text)
        real_words.append(row['source_text'].split(':')[1].split())
        if mode['slot_filling'] and mode['intent_detection']:
            predicted_slots.append(result_text.split(';')[1].split())
            real_slots.append(row['target_text'].split(';')[1].split())
            predicted_intents.append(result_text.split(';')[0])
            real_intents.append(row['target_text'].split(';')[0])

        elif mode['slot_filling']:
            predicted_slots.append(result_text.split())
            real_slots.append(row['target_text'].split())

            # predicted_slots.append([w.split(':')[1].strip() if len(w.split(':'))>1 else w.strip() for w in result_text.split(',')])
            # real_slots.append([w.split(':')[1].strip() if len(w.split(':'))>1 else w.strip() for w in row['target_text'].split(',')])

        elif mode['intent_detection']:
            predicted_intents.append(result_text)
            real_intents.append(row['target_text'])

    write_results(os.path.join('T5', 'result', f"T5_result_{mode['intent_detection']}_{mode['slot_filling']}.txt"), test_data['source_text'], test_data['target_text'], predicted_text)
    print("result data is saved in 'T5/result/T5_result.txt'\n")
    show_scores(real_words, mode, predicted_slots, predicted_intents, real_slots, real_intents)
