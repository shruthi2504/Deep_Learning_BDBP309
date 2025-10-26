#one code to run LSTM or GRU based on our choice - change the cell type to GRU to run GRU
import os
import re
import string
from string import digits
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, Embedding, Dense
from tensorflow.keras.models import Model
import time


DATA_PATH = "Hindi_English_Truncated_Corpus.csv"   # path to your CSV
SAMPLE_SIZE = 25000
MAX_SENT_LEN = 20
BATCH_SIZE = 64
EPOCHS = 10
LATENT_DIM = 256
CELL_TYPE = 'GRU'    # 'LSTM' or 'GRU'
NUM_TEST_SAMPLES = 200
OVERFIT_SANITY = False

# --------------------------
#cleaning
# --------------------------
def clean_text(s):
    s = str(s).lower()
    s = re.sub("'", '', s)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    s = s.translate(str.maketrans('', '', digits))
    s = re.sub("[२३०८१५७९४६]", "", s)
    s = re.sub(" +", " ", s.strip())
    return s

# --------------------------
# Load and preprocess
# --------------------------
def load_and_filter(path, sample_size=SAMPLE_SIZE):
    df = pd.read_csv(path, encoding='utf-8')
    df = df[df['source'] == 'ted']
    df = df[~pd.isnull(df['english_sentence'])]
    df.drop_duplicates(inplace=True)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    df['english_sentence'] = df['english_sentence'].astype(str).apply(clean_text)
    df['hindi_sentence'] = df['hindi_sentence'].astype(str).apply(clean_text)
    df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: 'START_ ' + x + ' _END')
    df['len_en'] = df['english_sentence'].apply(lambda x: len(x.split()))
    df['len_hi'] = df['hindi_sentence'].apply(lambda x: len(x.split()))
    df = df[(df['len_en'] <= MAX_SENT_LEN) & (df['len_hi'] <= MAX_SENT_LEN)]
    df = shuffle(df, random_state=42)
    df.reset_index(drop=True, inplace=True)
    return df

# --------------------------
# Prepare vocab
# --------------------------
def prepare_vocab(lines):
    all_eng_words = set(word for sentence in lines['english_sentence'] for word in sentence.split())
    all_hindi_words = set(word for sentence in lines['hindi_sentence'] for word in sentence.split())

    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_hindi_words))

    # +1 to reserve 0 for padding (and allow max index)
    num_encoder_tokens = len(all_eng_words) + 1
    num_decoder_tokens = len(all_hindi_words) + 1

    input_token_index = {word: i + 1 for i, word in enumerate(input_words)}
    target_token_index = {word: i + 1 for i, word in enumerate(target_words)}

    reverse_input_char_index = {i: word for word, i in input_token_index.items()}
    reverse_target_char_index = {i: word for word, i in target_token_index.items()}

    # fix: use correct column names
    max_length_src = max(lines['len_en'])
    max_length_tar = max(lines['len_hi'])

    return (num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index,
            reverse_input_char_index, reverse_target_char_index, max_length_src, max_length_tar)

# --------------------------
# Batch generator
# --------------------------
def data_generator(encoder_texts, decoder_texts, batch_size, max_length_src, max_length_tar,
                   input_token_index, target_token_index, num_decoder_tokens):
    n = len(encoder_texts)
    i = 0
    while True:
        encoder_input_data = np.zeros((batch_size, max_length_src), dtype='int32')
        decoder_input_data = np.zeros((batch_size, max_length_tar), dtype='int32')
        decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens), dtype='float32')

        for b in range(batch_size):
            if i >= n:
                i = 0
            src = encoder_texts.iloc[i]
            tgt = decoder_texts.iloc[i]
            # encoder
            for t, word in enumerate(src.split()):
                if t < max_length_src:
                    encoder_input_data[b, t] = input_token_index.get(word, 0)
            # decoder input and target (one-hot) (teacher forcing)
            tokens = tgt.split()
            for t, word in enumerate(tokens):
                if t < max_length_tar:
                    # decoder input gets the token (except last)
                    decoder_input_data[b, t] = target_token_index.get(word, 0)
                if t > 0 and (t - 1) < max_length_tar:
                    # decoder target is next token (one-hot)
                    idx = target_token_index.get(word, 0)
                    if idx < num_decoder_tokens:
                        decoder_target_data[b, t - 1, idx] = 1.0
            i += 1

        # yield inputs as a tuple ((enc_in, dec_in), dec_target)
        yield (encoder_input_data, decoder_input_data), decoder_target_data

# --------------------------
# Build seq2seq model (LSTM or GRU)
# --------------------------
def build_model(num_encoder_tokens, num_decoder_tokens, latent_dim, cell_type='LSTM'):
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(input_dim=num_encoder_tokens, output_dim=latent_dim, mask_zero=True, name='enc_emb')(encoder_inputs)
    if cell_type == 'LSTM':
        encoder_cell = LSTM(latent_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_cell(enc_emb)
        encoder_states = [state_h, state_c]
    else:
        encoder_cell = GRU(latent_dim, return_state=True, name='encoder_gru')
        encoder_outputs, state_h = encoder_cell(enc_emb)
        encoder_states = [state_h]  # GRU single state

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb_layer = Embedding(input_dim=num_decoder_tokens, output_dim=latent_dim, mask_zero=True, name='dec_emb')
    dec_emb = dec_emb_layer(decoder_inputs)
    if cell_type == 'LSTM':
        decoder_cell = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_cell(dec_emb, initial_state=encoder_states)
    else:
        decoder_cell = GRU(latent_dim, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_outputs, _ = decoder_cell(dec_emb, initial_state=encoder_states)

    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model, encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_cell, decoder_dense

# --------------------------
# Build inference models
# --------------------------
def build_inference_models(encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_cell, decoder_dense, latent_dim, cell_type='LSTM'):
    # Encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder inference model
    if cell_type == 'LSTM':
        decoder_state_input_h = Input(shape=(latent_dim,), name='dec_state_h')
        decoder_state_input_c = Input(shape=(latent_dim,), name='dec_state_c')
        dec_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        dec_emb2 = dec_emb_layer(decoder_inputs)
        decoder_outputs2, state_h2, state_c2 = decoder_cell(dec_emb2, initial_state=dec_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        decoder_model = Model([decoder_inputs] + dec_states_inputs, [decoder_outputs2] + decoder_states2)
    else:
        decoder_state_input_h = Input(shape=(latent_dim,), name='dec_state_h')
        dec_states_inputs = [decoder_state_input_h]

        dec_emb2 = dec_emb_layer(decoder_inputs)
        decoder_outputs2, state_h2 = decoder_cell(dec_emb2, initial_state=dec_states_inputs)
        decoder_states2 = [state_h2]
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        decoder_model = Model([decoder_inputs] + dec_states_inputs, [decoder_outputs2] + decoder_states2)

    return encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_dec_len, cell_type='LSTM'):
    # Encode
    states_value = encoder_model.predict(input_seq)

    # Prepare start token
    start_idx = target_token_index.get('START_', None)
    if start_idx is None:
        start_idx = target_token_index.get('start_', 1)

    target_seq = np.zeros((1, 1), dtype='int32')
    target_seq[0, 0] = start_idx

    decoded_tokens = []
    current_states = states_value

    for _ in range(max_dec_len):
        # Build inputs depending on whether states are list or single
        inputs = [target_seq] + (current_states if isinstance(current_states, list) else [current_states])
        outputs_and_states = decoder_model.predict(inputs)
        output_tokens = outputs_and_states[0]  # softmax over vocab
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        sampled_word = reverse_target_char_index.get(sampled_token_index, '')

        if sampled_word in ('_END', '_end'):
            break
        if sampled_word:
            decoded_tokens.append(sampled_word)

        # update target_seq and states
        target_seq = np.zeros((1, 1), dtype='int32')
        target_seq[0, 0] = sampled_token_index
        current_states = outputs_and_states[1:]  # rest are states

    return ' '.join(decoded_tokens)

# --------------------------
# Main
# --------------------------
def main(cell_type=CELL_TYPE):
    print("Loading and preprocessing data...")
    df = load_and_filter(DATA_PATH, SAMPLE_SIZE)
    (num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index,
     reverse_input_char_index, reverse_target_char_index, max_length_src, max_length_tar) = prepare_vocab(df)

    print(f"Vocab sizes -> encoder: {num_encoder_tokens}  decoder: {num_decoder_tokens}")
    print(f"Max lengths -> src: {max_length_src}  tar: {max_length_tar}")

    # quick debug: ensure START_ exists
    print('START_ in target vocab?', 'START_' in target_token_index)

    # Optionally try to overfit small set to verify correctness
    if OVERFIT_SANITY:
        print("Running overfit sanity test on 200 samples...")
        small = df.sample(n=min(200, len(df)), random_state=1)
        Xs = small['english_sentence']
        Ys = small['hindi_sentence']
        # build model
        model, encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_cell, decoder_dense = build_model(
            num_encoder_tokens, num_decoder_tokens, LATENT_DIM, cell_type=cell_type)
        train_gen = data_generator(Xs, Ys, batch_size=32, max_length_src=max_length_src, max_length_tar=max_length_tar,
                                   input_token_index=input_token_index, target_token_index=target_token_index,
                                   num_decoder_tokens=num_decoder_tokens)
        model.fit(train_gen, steps_per_epoch=len(Xs)//32 or 1, epochs=50)
        print("Overfit test done. If loss did not drop strongly, debug generator/token mapping.")
        return

    # Train/test split
    X, y = df['english_sentence'], df['hindi_sentence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # build model
    print(f"Building {cell_type} model...")
    model, encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_cell, decoder_dense = build_model(
        num_encoder_tokens, num_decoder_tokens, LATENT_DIM, cell_type=cell_type)
    model.summary()
    print("Total params:", model.count_params())

    # generate one batch and inspect
    gen = data_generator(X_train, y_train, batch_size=2, max_length_src=max_length_src, max_length_tar=max_length_tar,
                         input_token_index=input_token_index, target_token_index=target_token_index,
                         num_decoder_tokens=num_decoder_tokens)
    (enc_b, dec_b), dec_t = next(gen)
    print("Sample encoder batch shape:", enc_b.shape, "sample:", enc_b[0][:10])
    print("Sample decoder batch shape:", dec_b.shape, "sample:", dec_b[0][:10])
    print("Sample decoder target shape(one-hot):", dec_t.shape)

    # Train using generator
    train_gen = data_generator(X_train, y_train, batch_size=BATCH_SIZE, max_length_src=max_length_src, max_length_tar=max_length_tar,
                               input_token_index=input_token_index, target_token_index=target_token_index,
                               num_decoder_tokens=num_decoder_tokens)
    val_gen = data_generator(X_test, y_test, batch_size=BATCH_SIZE, max_length_src=max_length_src, max_length_tar=max_length_tar,
                             input_token_index=input_token_index, target_token_index=target_token_index,
                             num_decoder_tokens=num_decoder_tokens)

    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    validation_steps = max(1, len(X_test) // BATCH_SIZE)

    print("Starting training...")
    t0 = time.time()
    hist = model.fit(train_gen,
                     steps_per_epoch=steps_per_epoch,
                     epochs=EPOCHS,
                     validation_data=val_gen,
                     validation_steps=validation_steps)
    print("Training finished in %.1f sec" % (time.time() - t0))

    # Build inference models
    # --------------------------
    # Build inference models
    # --------------------------
    # Include latent_dim in model_info and pass properly
    model_info = (encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_cell, decoder_dense)

    encoder_model, decoder_model = build_inference_models(
        *model_info, latent_dim=LATENT_DIM, cell_type=cell_type
    )

    # Decode 5 examples from the train set
    print("\n--- Example translations (greedy) ---")
    single_gen = data_generator(
        X_train, y_train, batch_size=1,
        max_length_src=max_length_src, max_length_tar=max_length_tar,
        input_token_index=input_token_index,
        target_token_index=target_token_index,
        num_decoder_tokens=num_decoder_tokens
    )

    for i in range(5):
        (enc_seq, dec_seq), _ = next(single_gen)
        decoded = decode_sequence(
            enc_seq, encoder_model, decoder_model,
            target_token_index, reverse_target_char_index,
            max_dec_len=max_length_tar, cell_type=cell_type
        )
        print("Src:", X_train.iloc[i])
        print("Ref:", y_train.iloc[i].replace('START_ ', '').replace(' _END', ''))
        print("Pred:", decoded)
        print('-' * 40)

    return hist, model, encoder_model, decoder_model

if __name__ == '__main__':
    main(cell_type=CELL_TYPE)
