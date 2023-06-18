from transformers.models.bert.modeling_bert  import BertConfig
from transformers import BertTokenizer
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle
import os
from model import NewBert
from result.eval import conlleval

def show_scores(words, predicted_slots, predicted_intents, real_slots, real_intents):
    slot_scores = conlleval(predicted_slots, real_slots, words, os.path.join('bert', 'result' 'temp_for_conlleval.txt'))
    intent_accuracy = accuracy_score(real_intents, predicted_intents)
    print(f"slot scores: [precision:{slot_scores['p']}, recall:{slot_scores['r']}, F1:{slot_scores['f1']}]")
    print("intent accuracy: {0:.3f}".format(intent_accuracy))

def get_results(model_output, info, texts):
    intent_logits, slot_logits = model_output
    intent_indexes = intent_logits.argmax(-1).tolist()
    slot_indexes = slot_logits.argmax(-1).tolist()
    slots = []
    intents = []
    for i in range(len(intent_indexes)):
        slots.append([info['id2slot'][s] for s in slot_indexes[i][:len(texts[i].split())]])
        intents.append(info['id2intent'][intent_indexes[i]])
    return slots, intents

def write_results(path, texts, slots, intents, real_slots, real_intents):
    f = open(path, 'w')
    for i in range(len(texts)):
        f.write('text: ' + str(texts[i]) + '\n')
        f.write('real slots: ' + str(real_slots[i]) + '\n')
        f.write('predicted slots: ' + str(slots[i]) + '\n')
        f.write('real intent: ' + str(real_intents[i]) + '\n')
        f.write('predicted intent: ' + str(intents[i]) + '\n')
        f.write('#########################\n')
    f.close()


def evaluate(model, test_dataloader, info, tokenizer, device):
    print('using ' + str(device))
    loop = tqdm(test_dataloader, leave=True)
    model.eval()
    test_loss = []
    all_texts = []
    all_words = []
    all_slots = []
    all_intents = []
    real_slots = []
    real_intents = []
    for step, batch in enumerate(loop):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device) 
        slots_ids = batch[3].to(device)
        intent_ids = batch[4].to(device)
        with torch.no_grad(): 
            outputs = model(input_ids, attention_mask, token_type_ids, intent_ids, slots_ids)
        loss = outputs[1]
        test_loss.append(loss.item())
        avg_loss = sum(test_loss)/len(test_loss)
        loop.set_description('testing')
        loop.set_postfix(loss=avg_loss)

        ## convert results to text
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        slots, intents = get_results(outputs[0], info, texts)

        real_slot_ids = slots_ids.tolist()
        real_intent_ids = intent_ids.tolist()
        for i in range(len(intent_ids)):
            real_slots.append([info['id2slot'][s] for s in real_slot_ids[i] if info['id2slot'][s] != 'pad'])
            real_intents.append(info['id2intent'][real_intent_ids[i]])

        all_texts.extend(texts)
        all_words.extend([t.split() for t in texts])
        all_slots.extend(slots)
        all_intents.extend(intents)

    write_results(os.path.join('bert', 'result', 'bert_result.txt'), all_texts, all_slots, all_intents, real_intents, real_slots)
    print("result data is saved in 'bert/result/bert_result.txt'\n")
    print("evaluation scores:")
    show_scores(all_words, all_slots, all_intents, real_slots, real_intents)


if __name__ == "__main__":
    print("load test data..")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with open(os.path.join('bert', 'data', 'test_dataloader.pth'), "rb") as file:
        test_dataloader = pickle.load(file)
    with open(os.path.join('bert', 'data', 'info.p'), 'rb') as file:
        info = pickle.load(file)

    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device: ' + str(device))
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = NewBert(config, len(info['intent2id']), len(info['slot2id']), 0.1)
    model.load_state_dict(torch.load(os.path.join('bert', 'model', 'bert.pt'), map_location=torch.device(device)))
    model.to(device)
    evaluate(model, test_dataloader, info, tokenizer, device)
