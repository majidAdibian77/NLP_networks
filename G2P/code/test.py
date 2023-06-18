import argparse
import os
import pickle

from utils.test import prepare_test_data, test, word_error_rate, phoneme_error_rate
from utils.encoder_decoder_model import restore_model as restore_encoder_decoder_model
from utils.transformer_model import restore_model as restore_transformer_model

def save_predictioins(path, predicted_phonems, test_labels, graphemes):
    lines = []
    for i in range(len(predicted_phonems)):
        line = '\t'.join([graphemes[i], test_labels[i], predicted_phonems[i]])
        lines.append(line + '\n')
    with open(path, 'w') as f:
        f.writelines(lines)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n_gram', required=True)
    parser.add_argument('--verbose', default=False, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.data_path.split('/')[0], '{}_{}_info.p'.format(args.model, args.n_gram)), 'rb') as f:
        info = pickle.load(f)
    if args.model == 'encoder_decoder':
        model = restore_encoder_decoder_model(os.path.join('models', args.model, 'encoder_decoder_{}.tf'.format(args.n_gram)), info)
    else:
        model = restore_transformer_model(os.path.join('models', args.model, 'transformer_{}.h5'.format(args.n_gram)), info)
    test_data, test_labels, graphemes = prepare_test_data(args.data_path, info)
    predicted_phonems = test(model, args.model, test_data, graphemes, info, verbose=args.verbose=='True') 
    WER = word_error_rate(test_labels, predicted_phonems)
    PER = phoneme_error_rate(test_labels, predicted_phonems)
    result_path = os.path.join('.', 'results')
    os.makedirs(result_path, exist_ok = True)
    result_path = os.path.join(result_path, '{}.txt'.format(args.model))
    save_predictioins(result_path, predicted_phonems, test_labels, graphemes)
    print('WER: {0} ,  PER: {1}'.format(WER, PER))

if __name__ == "__main__":
    main()