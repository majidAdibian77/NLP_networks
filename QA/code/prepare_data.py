import datasets
import json
import random

def read_data(path):
    f = open(path, encoding='UTF-8')
    data = json.load(f)['data']
    all_context, all_questions, all_answers, all_answer_indexes = [], [], [], []
    for part in data:
        for paragraph in part['paragraphs']:
            context = paragraph['context']
            for qas in paragraph['qas']:
                question = qas['question']
                if not qas['is_impossible']:
                    for answer in qas['answers']:
                        all_answers.append(answer['text'])
                        all_answer_indexes.append(answer['answer_start'])
                        all_questions.append(question)
                        all_context.append(context)
                else:
                    all_answers.append("")
                    all_answer_indexes.append(-1)
                    all_questions.append(question)
                    all_context.append(context)
    data = all_context, all_questions, all_answers, all_answer_indexes
    return data

def read_and_add_squad_data(path, train_data, val_data):
    all_context, all_questions, all_answers, all_answer_indexes = [], [], [], []
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    for part in squad_dict['data']:
        for passage in part['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if not qa['is_impossible']:
                    for answer in qa['answers']:
                        all_context.append(context)
                        all_questions.append(question)
                        all_answers.append(answer['text'])
                        all_answer_indexes.append(answer['answer_start'])
                else:
                    all_context.append(context)
                    all_questions.append(question)
                    all_answers.append('')
                    all_answer_indexes.append(-1)
    train_context, train_questions, train_answers, train_answer_indexes = train_data
    val_context, val_questions, val_answers, val_answer_indexes = val_data

    train_data = all_context[8000:]+train_context, all_questions[8000:]+train_questions, \
                    all_answers[8000:]+train_answers, all_answer_indexes[8000:]+train_answer_indexes
    val_data = all_context[8000:]+val_context, all_questions[8000:]+val_questions, \
                    all_answers[8000:]+val_answers, all_answer_indexes[8000:]+val_answer_indexes
    train_data = list(zip(*train_data))
    random.shuffle(train_data)
    train_context, train_questions, train_answers, train_answer_indexes = zip(*train_data)
    train_data = train_context, train_questions, train_answers, train_answer_indexes
    val_data = list(zip(*val_data))
    random.shuffle(val_data)
    val_context, val_questions, val_answers, val_answer_indexes = zip(*val_data)
    val_data = val_context, val_questions, val_answers, val_answer_indexes
    return train_data, val_data

def tokenize_data(tokenizer, data, max_length, doc_stride):
    all_context, all_questions, all_answers, all_answer_indexes = data
    # all_context, all_questions, all_answers, all_answer_indexes = all_context[:10], all_questions[:10], all_answers[:10], all_answer_indexes[:10]
    tokenized_examples = tokenizer(
        all_questions,
        all_context,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,)
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answer = all_answers[sample_index]
        answer_index = all_answer_indexes[sample_index]
        if len(answer) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answer_index
            end_char = start_char + len(answer)
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
                
    dataset = datasets.Dataset.from_dict(tokenized_examples)
    return dataset

def read_test_data(path):
    f = open(path, encoding='UTF-8')
    data = json.load(f)['data']
    all_context, all_questions, all_answers, all_answer_indexes = [], [], [], []
    for part in data:
        for paragraph in part['paragraphs']:
            context = paragraph['context']
            for qas in paragraph['qas']:
                question = qas['question']
                if not qas['is_impossible']:
                    answers = []
                    for answer in qas['answers']:
                        answers.append({'text':answer['text'], 'answer_start':answer['answer_start']})
                    all_answers.append(answers)
                    all_questions.append(question)
                    all_context.append(context)
                else:
                    all_answers.append([{'text':"", 'answer_start':-1}])
                    all_answer_indexes.append(-1)
                    all_questions.append(question)
                    all_context.append(context)
    return all_context, all_questions, all_answers

def read_and_add_squad_test_data(path, test_data):
    all_context, all_questions, all_answers, all_answer_indexes = test_data
    all_new_answers = []
    for i in range(len(all_answers)):
        all_new_answers.append({'text':all_answers[i], 'answer_start':all_answer_indexes[i]})

    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    for part in squad_dict['data']:
        for passage in part['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if not qa['is_impossible']:
                    answers = []
                    for answer in qa['answers']:
                        answers.append({'text':answer['text'], 'answer_start':answer['answer_start']})
                    all_context.append(context)
                    all_questions.append(question)
                    all_new_answers.append(answers)
                else:
                    all_context.append(context)
                    all_questions.append(question)
                    all_new_answers.append([{'text':"", 'answer_start':-1}])
    return all_context, all_questions, all_new_answers, all_answer_indexes