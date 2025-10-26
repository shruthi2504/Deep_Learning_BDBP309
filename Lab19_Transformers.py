import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ----------------------
# Configuration
# ----------------------
IMAGES_PATH = "../Lab15/image_captioning_dataset/Images"
CAPTIONS_PATH = "../Lab15/image_captioning_dataset/captions.txt"
IMAGE_SIZE = (299, 299)
SEQ_LENGTH = 25
VOCAB_SIZE = 5000
EMBED_DIM = 256
FF_DIM = 256
BATCH_SIZE = 16
EPOCHS = 2
NUM_HEADS = 2

# ----------------------
# Load captions
# ----------------------
def load_captions(filename):
    with open(filename) as f:
        lines = f.readlines()[1:]
    caption_mapping, text_data = {}, []
    skip_images = set()
    for line in lines:
        img_name, caption = line.strip().split(",", 1)
        img_path = os.path.join(IMAGES_PATH, img_name)
        tokens = caption.strip().split()
        if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
            skip_images.add(img_path)
            continue
        caption = "<start> " + caption.strip() + " <end>"
        text_data.append(caption)
        caption_mapping.setdefault(img_path, []).append(caption)
    for img in skip_images:
        caption_mapping.pop(img, None)
    return caption_mapping, text_data

# ----------------------
# Split dataset
# ----------------------
def train_val_split(caption_mapping, val_size=0.2, test_size=0.05):
    images = list(caption_mapping.keys())
    train_keys, val_keys = train_test_split(images, test_size=val_size, random_state=42)
    val_keys, test_keys = train_test_split(val_keys, test_size=test_size, random_state=42)
    train_data = {k: caption_mapping[k] for k in train_keys}
    val_data = {k: caption_mapping[k] for k in val_keys}
    test_data = {k: caption_mapping[k] for k in test_keys}
    return train_data, val_data, test_data

# ----------------------
# Text vectorization
# ----------------------
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, r"[!\"#$%&'()*+,-./:;=?@[\]^_`{|}~0-9]", "")

def build_vectorizer(text_data):
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization
    )
    vectorizer.adapt(text_data)
    return vectorizer

# ----------------------
# Image preprocessing
# ----------------------
def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img

# ----------------------
# Dataset preparation
# ----------------------
def make_dataset(images, captions, vectorizer):
    imgs = [decode_and_resize(img) for img in images]
    cap_list = []
    for c in captions:
        cap_list.append(vectorizer(c[0]))  # take first caption
    dataset = tf.data.Dataset.from_tensor_slices((imgs, cap_list))
    dataset = dataset.shuffle(len(imgs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# ----------------------
# CNN model
# ----------------------
def get_cnn_model():
    base_model = keras.applications.EfficientNetB0(
        include_top=False, input_shape=(*IMAGE_SIZE,3), weights='imagenet'
    )
    base_model.trainable = False
    x = layers.Reshape((-1, base_model.output.shape[-1]))(base_model.output)
    return keras.Model(base_model.input, x)

# ----------------------
# Transformer blocks
# ----------------------
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        attn_out = self.mha(x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(seq_len, embed_dim)
        self.scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return self.token_emb(x)*self.scale + self.pos_emb(positions)

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, seq_len, vocab_size):
        super().__init__()
        self.embed = PositionalEmbedding(seq_len, vocab_size, embed_dim)
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.out = layers.Dense(vocab_size, activation="softmax")

    def call(self, x, enc_out):
        x = self.embed(x)
        attn1 = self.mha1(x, x)
        x = self.norm1(x + attn1)
        attn2 = self.mha2(x, enc_out)
        x = self.norm2(x + attn2)
        x = self.ffn(x)
        x = self.norm3(x + x)
        return self.out(x)

# ----------------------
# Caption Generation
# ----------------------
def generate_caption(model_parts, image_path, vectorizer, index_to_word):
    cnn, encoder, decoder = model_parts
    img = tf.expand_dims(decode_and_resize(image_path), 0)
    enc_out = encoder(cnn(img))
    dec_input = tf.expand_dims(vectorizer("<start>"), 0)

    for _ in range(SEQ_LENGTH):
        predictions = decoder(dec_input, enc_out)
        predicted_id = tf.argmax(predictions[:, -1, :], axis=-1)
        predicted_word = index_to_word[int(predicted_id)]
        if predicted_word == "<end>":
            break
        dec_input = tf.concat([dec_input, tf.expand_dims(predicted_id, 0)], axis=-1)
    caption_tokens = dec_input.numpy()[0]
    caption = " ".join([index_to_word[int(i)] for i in caption_tokens])
    return caption.replace("<start>", "").replace("<end>", "").strip()

# ----------------------
# Main
# ----------------------
def main():
    # BLEU smoothing function
    smooth_fn = SmoothingFunction().method1

    captions_mapping, text_data = load_captions(CAPTIONS_PATH)
    train_data, val_data, test_data = train_val_split(captions_mapping)
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    vectorizer = build_vectorizer(text_data)
    vocab = vectorizer.get_vocabulary()
    index_to_word = {i: w for i, w in enumerate(vocab)}

    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()), vectorizer)
    val_dataset = make_dataset(list(val_data.keys()), list(val_data.values()), vectorizer)

    cnn = get_cnn_model()
    encoder = TransformerEncoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)
    decoder = TransformerDecoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS, SEQ_LENGTH, VOCAB_SIZE)

    optimizer = keras.optimizers.Adam(1e-4)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for imgs, caps in train_dataset:
            with tf.GradientTape() as tape:
                enc_out = encoder(cnn(imgs))
                dec_out = decoder(caps[:, :-1], enc_out)
                loss = loss_fn(caps[:, 1:], dec_out)
            grads = tape.gradient(loss, cnn.trainable_variables + encoder.trainable_variables + decoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, cnn.trainable_variables + encoder.trainable_variables + decoder.trainable_variables))
            print(f"Batch loss: {loss.numpy():.4f}", end='\r')

        # Evaluate using BLEU on validation images
        references, hypotheses = [], []
        for img_path, caps in list(val_data.items())[:50]:  # small subset for CPU
            pred_caption = generate_caption((cnn, encoder, decoder), img_path, vectorizer, index_to_word)
            references.append([caps[0].replace("<start>", "").replace("<end>", "").split()])
            hypotheses.append(pred_caption.split())
        bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn)
        print(f"\nEpoch {epoch+1} Validation BLEU Score: {bleu:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
