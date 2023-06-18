from jiwer import cer, wer
import numpy as np
import tensorflow as tf

def prpare_test_sample(grapheme, info):
    sample_data = np.zeros([1, info['max_len_grapheme']], dtype=np.uint32)
    if info['encoder_ngram'] == 'bigram':
        #### bigram
        sample_data[0, 0] = info['grapheme2id']['<bos>'+grapheme[0]]
        for j in range(1, len(grapheme)):
            sample_data[0, j] = info['grapheme2id'][grapheme[j-1]+grapheme[j]]
        sample_data[0][len(grapheme)] = info['grapheme2id'][grapheme[len(grapheme)-1]+'<eos>']
        ####
    else:
        for j, c in enumerate(grapheme):
            sample_data[0, j] = info['grapheme2id'][c]
        sample_data[0][len(grapheme)] = info['grapheme2id']['<eos>']
    return sample_data
    
def prepare_test_data(path, info):
    print('reading data ...')
    f = open(path, 'r')
    lines = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines)
    lines = lines[:1000]
    f.close()
    data = np.zeros([len(lines), info['max_len_grapheme']], dtype=np.uint16)
    labels = []
    graphemes = []
    for i, line in enumerate(lines):
        grapheme, phoneme = line.strip().split('\t')
        graphemes.append(grapheme)
        labels.append(phoneme)
        data[i] = prpare_test_sample(grapheme, info)
    return data, labels, graphemes

def word_error_rate(real_phonems, predicted_phonems):
    WER = wer(real_phonems, predicted_phonems)
    return WER

def phoneme_error_rate(real_phonems, predicted_phonems):
    PER = cer(real_phonems, predicted_phonems)
    return PER


def G2P_sample_encoder_decoder(model, test_sample, info):
    V_or_C = info['V_or_C']
    CVs = []
    phonemes = []

    encoder_model, decoder_model, attention_model = model['encoder'], model['decoder'], model['attention']
    encoder_output, states_value = encoder_model.predict(test_sample, verbose=0)
    target_seq = np.zeros((1, 1), dtype="int8")
    target_seq[0, 0] = info['phoneme2id']['<bos>']
    phonemes = []
    for _ in range(info['max_len_phoneme']):
        attention_output = attention_model.predict([target_seq, encoder_output], verbose=0)
        predictions, states_value = decoder_model.predict([attention_output, states_value], verbose=0)
        predictions = predictions[0, -1, :]
        _, indices = tf.nn.top_k(predictions, 5)
        indices = indices.numpy().flatten ()
        if CVs:
            for i in range(indices.shape[0]):
                ph_CV = V_or_C[info['id2phoneme'][indices[i]]]
                if ph_CV == 'E':
                    predicted_id = indices[i]
                    break
                else:
                    if (CVs[-1]=='C' and ph_CV=='V') or (CVs[-1]=='V' and ph_CV=='C') or (CVs[-1]=='C' and ph_CV=='C'):
                        if len(phonemes)<2 or info['id2phoneme'][indices[i]] != phonemes[-1] or (info['id2phoneme'][indices[i]] == phonemes[-1] and phonemes[-1] != phonemes[-2]):
                            predicted_id = indices[i]
                            break
        else:
            predicted_id = np.argmax(predictions)


        phoneme = info['id2phoneme'][predicted_id]
        CVs.append(V_or_C[phoneme])
        if phoneme == '<eos>' or phoneme == '<pad>':
            break
        phonemes.append(phoneme)
        target_seq = np.concatenate([target_seq, predicted_id.reshape((1, 1))], axis=-1)

    res =  "".join([phoneme for phoneme in phonemes if phoneme not in ['<bos>', '<eos>', '<pad>']])
    if len(res) >= 6:
        for i in range(len(res)):
            if res[i]+res[i-1] == res[i-2]+res[i-3] and res[i]+res[i-1] == res[i-4]+res[i-5]:
                res = res[:i-5]
                break
    return res

def G2P_sample_transformer(model, test_sample, info):
    V_or_C = info['V_or_C']
    CVs = []
    phonemes = []

    output = np.zeros((1, 1), dtype="int8")
    output[0, 0] = info['phoneme2id']['<bos>']
    for i in range(info['max_len_phoneme']):
        predictions = model(inputs=[test_sample, output], training=False)
        predictions = predictions[:, -1:, :]

        _, indices = tf.nn.top_k(predictions, 5)
        indices = indices.numpy().flatten ()
        if CVs:
            for i in range(indices.shape[0]):
                ph_CV = V_or_C[info['id2phoneme'][indices[i]]]
                if ph_CV == 'E':
                    predicted_id = indices[i]
                    break
                else:
                    if (CVs[-1]=='C' and ph_CV=='V') or (CVs[-1]=='V' and ph_CV=='C') or (CVs[-1]=='C' and ph_CV=='C'):
                        if info['id2phoneme'][indices[i]] != phonemes[-1] or (info['id2phoneme'][indices[i]] == phonemes[-1] and phonemes[-1] != phonemes[-2]):
                            predicted_id = indices[i]
                            predicted_id_as_tf = tf.cast(predicted_id.reshape((1,1)), tf.int8)
                            break
        else:
            predicted_id_as_tf = tf.cast(tf.argmax(predictions, axis=-1), tf.int8)
            predicted_id = predicted_id_as_tf.numpy().item()

        phoneme = info['id2phoneme'][predicted_id]
        CVs.append(V_or_C[phoneme])
        if phoneme == '<eos>':
            break
        phonemes.append(phoneme)
        output = tf.concat([output, predicted_id_as_tf], axis=-1)

    res =  "".join([phoneme for phoneme in phonemes if phoneme not in ['<bos>', '<eos>', '<pad>']])
    if len(res) >= 6:
        for i in range(len(res)):
            if res[i]+res[i-1] == res[i-2]+res[i-3] and res[i]+res[i-1] == res[i-4]+res[i-5]:
                res = res[:i-5]
                break
    return res

def test(model, model_type, test_data, graphemes, info, verbose=False):
    predicted_phonems = []
    for i, test_samples in enumerate(test_data):
        if model_type == 'encoder_decoder':
            phonem = G2P_sample_encoder_decoder(model, np.expand_dims(test_samples, axis=0), info)
        else:
            phonem = G2P_sample_transformer(model, np.expand_dims(test_samples, axis=0), info)
        predicted_phonems.append(phonem)
        if verbose:
            print('grapheme input: ' + graphemes[i])
            print('model outpu: ' + phonem)
            print()
    return predicted_phonems