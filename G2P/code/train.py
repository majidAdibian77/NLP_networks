import argparse
import os
import pickle
import matplotlib.pyplot as plt

from utils.encoder_decoder_model import build_model as build_encoder_decoder_model
from utils.transformer_model import build_model as build_transformer_model
from utils.data import prepare_data

def plot_losses(losses, path, title):
    plt.plot(losses['train_loss'])
    plt.plot(losses['val_loss'])
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(path)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_path', default='models/', required=True)
    parser.add_argument('--data_path', default='data/', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n_gram', default='bigram', required=True)
    parser.add_argument('--batch_size', default=32, required=True)
    parser.add_argument('--val_size', default=0.01, required=True)
    args = parser.parse_args()

    train_dataset, val_dataset, info = prepare_data(args.data_path, int(args.batch_size), args.model, args.n_gram, float(args.val_size))
    with open(os.path.join(args.data_path.split('/')[0], '{}_{}_info.p'.format(args.model, args.n_gram)), 'wb') as f:
        pickle.dump(info, f)
        f.close()

    print('building model..')
    losses = {'train_loss': [], 'val_loss': []}
    if args.model == 'encoder_decoder':
        model = build_encoder_decoder_model(encoder_vocab_size=len(info['grapheme2id']), decoder_vocab_size=len(info['phoneme2id']), info=info)
        print('start training..')
        epoch = 10
        for _ in range(epoch):
            history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)
            losses['train_loss'].append(history.history['loss'])
            losses['val_loss'].append(history.history['val_loss'])
            model_path = os.path.join(args.save_model_path, args.model)
            os.makedirs(model_path, exist_ok = True)
            model.save_weights(os.path.join(model_path, 'encoder_decoder_{}.tf'.format(args.n_gram)), save_format='tf')
    else:
        model = build_transformer_model(info)
        print('start training..')
        epoch = 10
        for _ in range(epoch):
            history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)
            losses['train_loss'].append(history.history['loss'])
            losses['val_loss'].append(history.history['val_loss'])
            model_path = os.path.join(args.save_model_path, args.model)
            os.makedirs(model_path, exist_ok = True)
            model.save_weights(os.path.join(model_path,'transformer_{}.h5'.format(args.n_gram)))
    print('model is trained.')
    plot_losses(losses, os.path.join(model_path, '{0}_{1}_loss.jpg'.format(args.model, args.n_gram)), 'train and val losses per epoch in ' + str(args.model) + ' using ' + str(args.n_gram))


if __name__ == "__main__":
    main()