import tensorflow as tf
tf.random.set_seed(1234)
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, \
    Embedding, Concatenate, Attention, LayerNormalization


class Encoder(tf.keras.Model):
    def __init__(self, encoder_vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder_embedded = Embedding(encoder_vocab_size, embed_dim)
        self.encoder_lstm1 = Bidirectional(LSTM(hidden_dim, return_sequences=True, return_state=False, dropout=0.1))
        self.encoder_lstm2 = Bidirectional(LSTM(hidden_dim, return_sequences=True, return_state=True, dropout=0.1))

    def call(self, encoderinput):
        encoder_embedded = self.encoder_embedded(encoderinput)
        lstm1_output = self.encoder_lstm1(encoder_embedded)
        encoder_output, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm2(lstm1_output)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        return encoder_output, encoder_states

class Decoder(tf.keras.Model):
    def __init__(self, decoder_vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.decoder_lstm = LSTM(2*hidden_dim, return_sequences=True, return_state=True)
        self.dence = Dense(decoder_vocab_size, activation='softmax')

    def call(self, input):
        attention_output, encoder_states = input[0], input[1]
        logits, state_h_dec, state_c_dec= self.decoder_lstm(attention_output, initial_state=encoder_states)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_output = self.dence(logits)
        return decoder_output, decoder_states

class Attention_layer(tf.keras.Model):
    def __init__(self, decoder_vocab_size, embed_dim):
        super(Attention_layer, self).__init__()
        self.decoder_embedded = Embedding(decoder_vocab_size, embed_dim)
        self.attention = Attention()
        self.dence_q = Dense(embed_dim)
        self.dence_k = Dense(embed_dim)
        self.dence_v = Dense(embed_dim)
        self.normalization = LayerNormalization()

    def call(self, input):
        decoder_input, encoder_output = input[0], input[1]
        decoder_embedded = self.decoder_embedded(decoder_input)
        query = self.dence_q(decoder_embedded)
        key = self.dence_k(encoder_output)
        value = self.dence_v(decoder_embedded)
        attention_sequence = self.attention(inputs=[query, key])
        attention_output = self.normalization(value + attention_sequence)
        return attention_output

class Encoder_Decoder(tf.keras.Model):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_dim, hidden_dim):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(encoder_vocab_size, embed_dim, hidden_dim)
        self.attention = Attention_layer(decoder_vocab_size, embed_dim)
        self.decoder = Decoder(decoder_vocab_size, embed_dim, hidden_dim)

    def call(self, input):
        encoder_input, decoder_input = input['input_1'], input['input_2']
        encoder_output, encoder_states = self.encoder(encoder_input)
        attention_output = self.attention([decoder_input, encoder_output])
        decoder_output, _ = self.decoder([attention_output, encoder_states])
        return decoder_output
        
def build_model(encoder_vocab_size, decoder_vocab_size, info, hidden_dim=256, embed_dim=128):
    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, info['max_len_phoneme']))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss) 
    encoder_decoder_model = Encoder_Decoder(encoder_vocab_size, decoder_vocab_size, embed_dim, hidden_dim)
    encoder_decoder_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return encoder_decoder_model

def restore_model(model_path, info, hidden_dim=256, embed_dim=128):
    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, info['max_len_phoneme']))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss) 
    model = build_model(encoder_vocab_size=len(info['grapheme2id']), decoder_vocab_size=len(info['phoneme2id']), info=info, hidden_dim=hidden_dim, embed_dim=embed_dim)
    model.compile(optimizer='adam', loss=loss_function)
    model.load_weights(model_path)
    model = {'encoder': model.encoder, 'decoder': model.decoder, 'attention': model.attention}
    return model