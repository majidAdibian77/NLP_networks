import tensorflow as tf
import numpy as np

def transformer_dataset(encoder_input, decoder_input, decoder_output, batch_size, val_size):
    encoder_input_train = encoder_input[int(val_size*encoder_input.shape[0]):]
    decoder_input_train = decoder_input[int(val_size*decoder_input.shape[0]):]
    decoder_output_train = decoder_output[int(val_size*decoder_output.shape[0]):]
    
    encoder_input_val = encoder_input[:int(val_size*encoder_input.shape[0])]
    decoder_input_val = decoder_input[:int(val_size*decoder_input.shape[0])]
    decoder_output_val = decoder_output[:int(val_size*decoder_output.shape[0])]

    BUFFER_SIZE = 20000
    dataset = tf.data.Dataset.from_tensor_slices((
        {'inputs': encoder_input_train, 'dec_inputs': decoder_input_train},
        {'outputs': decoder_output_train},
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    train_dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.from_tensor_slices((
        {'inputs': encoder_input_val, 'dec_inputs': decoder_input_val},
        {'outputs': decoder_output_val},
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    val_dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, val_dataset
    
    
def encoder_decoder_dataset(encoder_input, decoder_input, decoder_target, batch_size, val_size):
    BUFFER_SIZE = 20000
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_1': encoder_input,         
            'input_2': decoder_input
        },
        decoder_target
    ))
    val_size = int(float(val_size)*encoder_input.shape[0])
    val_data = dataset.take(val_size) 
    train_data = dataset.skip(val_size)
    train_and_val = []
    for dataset in [train_data, val_data]:
        dataset = dataset.cache()
        dataset = dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        train_and_val.append(dataset)
    return train_and_val[0], train_and_val[1]

def prepare_data(path, batch_size, encoder_decoder_of_transformer, n_gram, val_size=0.01):
    print('reading data ...')
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    data, labels = [], []
    grapheme_set, phoneme_set = set(), set()
    np.random.shuffle(lines)
    for line in lines:
        grapheme, phoneme = line.strip().split('\t')
        grapheme, phoneme = list(grapheme), list(phoneme)
        data.append(grapheme)
        labels.append(phoneme)
        grapheme_set.update(set(grapheme))
        phoneme_set.update(set(phoneme))
    
    if n_gram == 'bigram':
        #### bigram
        ngrapheme2id = {}
        bigrapheme_id = 0
        for grapheme1 in ['<pad>', '<unk>', '<bos>', '<eos>'] + list(grapheme_set):
            for grapheme2 in ['<pad>', '<unk>', '<bos>', '<eos>'] + list(grapheme_set):
                ngrapheme2id[grapheme1 + grapheme2] = bigrapheme_id
                bigrapheme_id += 1
        ####
    else:
        ngrapheme2id = {g:i for i,g in enumerate(['<pad>', '<unk>', '<bos>', '<eos>']+list(grapheme_set))}

    id2ngrapheme = {i:g for g,i in ngrapheme2id.items()}
    phoneme2id = {p:i for i,p in enumerate(['<pad>', '<unk>', '<bos>', '<eos>']+list(phoneme_set))}
    id2phoneme = {i:p for p,i in phoneme2id.items()}
    
    max_len_grapheme = len(max(data, key=lambda x:len(x)))
    max_len_phoneme = len(max(labels, key=lambda x:len(x)))

    print('creating matrixes ...')
    encoder_input = np.zeros([len(data), max_len_grapheme+1], dtype=np.uint16)
    decoder_input = np.zeros([len(data), max_len_phoneme+2], dtype=np.uint8)
    decoder_output = np.zeros([len(data), max_len_phoneme+2], dtype=np.uint8)

    for i in range(len(data)):
        if n_gram == 'bigram':
            #### bigram
            encoder_input[i][0] = ngrapheme2id['<bos>'+data[i][0]]
            for j in range(1, len(data[i])):
                encoder_input[i][j] = ngrapheme2id[data[i][j-1]+data[i][j]]
            encoder_input[i][len(data[i])] = ngrapheme2id[data[i][len(data[i])-1]+'<eos>']
            ####
        else:
            for j in range(len(data[i])):
                encoder_input[i][j] = ngrapheme2id[data[i][j]]
            encoder_input[i][len(data[i])] = ngrapheme2id['<eos>']
            
        decoder_input[i][0] = phoneme2id['<bos>']
        for j in range(len(labels[i])):
            decoder_input[i][j+1] = phoneme2id[labels[i][j]]
            decoder_output[i][j] = phoneme2id[labels[i][j]]
        decoder_input[i][len(labels[i])+1] = phoneme2id['<eos>']
        decoder_output[i][len(labels[i])] = phoneme2id['<eos>']
        
    V_or_C = {'A':'V', 'i':'V', 'e':'V', 'u':'V', 'a':'V', 'o':'V', '<eos>':'E', '<pad>':'E', '<bos>':'E', '<unk>':'E'}
    consonant = {ph:'C' for ph in phoneme2id.keys() if ph not in V_or_C.keys()}
    V_or_C.update(consonant)
    
    info = {'grapheme2id': ngrapheme2id, 'id2grapheme': id2ngrapheme, 'phoneme2id': phoneme2id, 
            'id2phoneme': id2phoneme, 'max_len_grapheme': encoder_input.shape[1], 'max_len_phoneme': decoder_input.shape[1],
            'batch_size': batch_size, 'encoder_ngram': n_gram, 'V_or_C': V_or_C}
    if encoder_decoder_of_transformer == 'encoder_decoder':
        train_dataset, val_dataset = encoder_decoder_dataset(encoder_input, decoder_input, decoder_output, batch_size, val_size)
    else:
        train_dataset, val_dataset = transformer_dataset(encoder_input, decoder_input, decoder_output, batch_size, val_size)
        
    print('train data is prepaired.')
    return train_dataset, val_dataset, info
