
import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import evaluate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from train import read_data
from prepare_data import read_test_data, read_and_add_squad_test_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predicted_and_ref(contexts, answers, best_starts_indexes, best_starts, best_ends_indexes, best_ends, tokens, tokenizer):
    references = []
    predictions = []
    for i in range(len(contexts)):
        context = contexts[i]
        min_null_score = best_starts_indexes[i][0] + best_ends_indexes[i][0]
        start_context = tokens['input_ids'][i].tolist().index(tokenizer.sep_token_id)

        offset = tokens['offset_mapping'][i]
        valid_answers = []
        for start_index in best_starts_indexes[i]:
            if start_index<start_context:
                continue
            for end_index in best_ends_indexes[i]:
                if (start_index >= len(offset) or end_index >= len(offset) or offset[start_index] is None or offset[end_index] is None):
                    continue
                if end_index < start_index:
                    continue
                start_char = offset[start_index][0]
                end_char = offset[end_index][1]
                valid_answers.append({"score": (best_starts[i][start_index] + best_ends[i][end_index]).item(), "text": context[start_char: end_char]})

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": min_null_score}

        predictions.append({
            'id': str(i),
            'prediction_text': best_answer['text'] if best_answer["score"] >= min_null_score else "",
            'no_answer_probability': min_null_score.item()/2
        })
        references.append({
            'id': str(i),
            'answers': answers[i],
        })
    return references, predictions
    
    
def get_best_starts_and_ends(model, questions, contexts, batch_size, device, n_best, tokenizer):
    tokens = tokenizer(questions, contexts, add_special_tokens=True,
                            return_token_type_ids=True, return_tensors="pt", padding=True,
                            return_offsets_mapping=True, truncation="only_second",
                            max_length=512, stride=256).to(device)
    best_starts_indexes, best_ends_indexes = [], []
    best_starts, best_ends = [], []
    for i in tqdm(range(0, len(questions), batch_size)):
        with torch.no_grad():
            out = model(tokens['input_ids'][i:i+batch_size].to(device),
                        tokens['attention_mask'][i:i+batch_size].to(device),
                        tokens['token_type_ids'][i:i+batch_size].to(device))
            start_logits = F.softmax(out.start_logits, dim=1)
            n_best_start_indexes = start_logits.argsort(dim=-1, descending=True)
            n_best_start_indexes = n_best_start_indexes[:, :n_best]
            end_logits = F.softmax(out.end_logits, dim=1)
            n_best_end_indexes = end_logits.argsort(dim=-1, descending=True)
            n_best_end_indexes = n_best_end_indexes[:, :n_best]
            best_starts.append(start_logits)
            best_ends.append(end_logits)
            best_starts_indexes.append(n_best_start_indexes)
            best_ends_indexes.append(n_best_end_indexes)
        
    tensor_best_starts_indexes = torch.cat(best_starts_indexes)
    tensor_best_ends_indexes = torch.cat(best_ends_indexes)
    tensor_best_starts = torch.cat(best_starts)
    tensor_best_ends = torch.cat(best_ends)
    return tensor_best_starts_indexes, tensor_best_ends_indexes, tensor_best_starts, tensor_best_ends, tokens
    
def eval_model(test_data, tokenizer, model):
    questions, contexts, answers = test_data
    tensor_best_starts_indexes, tensor_best_ends_indexes, tensor_best_starts, tensor_best_ends, tokens = \
        get_best_starts_and_ends(model, questions, contexts, 32, device, 20, tokenizer)
    references, predictions = get_predicted_and_ref(contexts, answers, tensor_best_starts_indexes, tensor_best_starts, \
                                                        tensor_best_ends_indexes, tensor_best_ends, tokens, tokenizer)
    metric = evaluate.load("squad_v2")
    eval_res = metric.compute(predictions=predictions, references=references)
    return eval_res
    
def show_example(test_data, example_number, tokenizer, model):
    contexts, questions, answers = test_data
    questions, contexts, answers = [questions[example_number]], [contexts[example_number]], [answers[example_number]]
    tensor_best_starts_indexes, tensor_best_ends_indexes, tensor_best_starts, tensor_best_ends, tokens = \
        get_best_starts_and_ends(model, questions, contexts, 16, device, 20, tokenizer)
    print(tensor_best_starts_indexes)
    print(tensor_best_ends_indexes)
    references, predictions = get_predicted_and_ref(contexts, answers, tensor_best_starts_indexes, tensor_best_starts, \
                                                        tensor_best_ends_indexes, tensor_best_ends, tokens, tokenizer)
    print("context:")
    print(contexts[0])
    print("question:")
    print(questions[0])
    print("refrence answer:")
    print(references[0]['answers'][0]['text'])
    print("predicted answer:")
    print(predictions[0]['prediction_text'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, help="type of model")
    parser.add_argument("--example_number", default=-1, type=int, help="just sjow one example or not")
    args = parser.parse_args()
    
    test_data = read_test_data(os.path.join('data/pquad','test.json'))
    if args.model_type == "pretrained_xlm_roberta":
        test_data2 = read_and_add_squad_test_data(os.path.join('data/squad','test.json'), test_data)
    model_checkpoint = os.path.join('models/', args.model_type, 'checkpoint-12000')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model = model.eval().to(device)
    if args.example_number == -1:
        if args.model_type == "pretrained_xlm_roberta":
            eval_res = eval_model(test_data2, tokenizer, model)
            print('results on test data that is contaned pcuad and squad test data:')
            print(eval_res)

            model = model.eval().to(device)
            eval_res = eval_model(test_data, tokenizer, model)
            print('results on test data that is contaned just pquad test data:')
            print(eval_res)
        else:
            model = model.eval().to(device)
            eval_res = eval_model(test_data, tokenizer, model)
            print('results on pquad test data:')
            print(eval_res)
    else:
        show_example(test_data, int(args.example_number), tokenizer, model)

if __name__ == "__main__":
    main()