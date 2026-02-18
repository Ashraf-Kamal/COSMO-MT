import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D,  GRU, Dense, Concatenate, Attention, Dropout, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics import classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

MAX_SEQ_LEN = 128  
EMBEDDING_DIM = 300  
VOCAB_SIZE = 600000 
FILTERS = 256
KERNEL_SIZE = 5 
DENSE_UNITS = 32
AUX_FEATURE_LEN = 15 

EPOCH = 30
BATCH_SIZE = 64

TEXT_COLUMN = "cleaned_text"
LABEL_COLUMN = "label" 

bert_model_path = "/Users/models/bert-base-uncased/"

datasets = ['Dataset-1','Dataset-2',]
data_name = datasets[1]

path = f"dataset/final_dataset/split_data/{data_name}"

df_train = pd.read_csv(path+"_train.csv")
df_test = pd.read_csv(path+"_test.csv")
df_val = pd.read_csv(path+"_val.csv")

print(df_train.shape)
print(df_test.shape)
print(df_val.shape)

df_train = df_train.dropna()
df_test = df_test.dropna()
df_val = df_val.dropna()

for column in df_val.columns:
    if df_val[column].isnull().any():
        print(f"Column '{column}' contains NaN values.")
    else:
        print(f"Column '{column}' does not contain NaN values.")

# Auxiliary Features
aux_path = f"dataset/Feature_Vectors (Normalised)_Results/{data_name}"
aux_df_train = pd.read_csv(aux_path+"_train_fvect27_all_final.csv")
aux_df_test = pd.read_csv(aux_path+"_test_fvect27_all_final.csv")
aux_df_val = pd.read_csv(aux_path+"_val_fvect27_all_final.csv")

# F1-Score 
def cust_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Proposed Model: COMO-MT

bert_model = TFBertModel.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

def bert_tokenize_texts(texts, tokenizer, max_len):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf"
    )
    return encodings['input_ids'], encodings['attention_mask']

bert_input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="BERT_Input_Ids")
bert_attention_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="BERT_Attention_Mask")

auxiliary_input = Input(shape=(AUX_FEATURE_LEN,), name="auxiliary_input")

def load_embedding_matrix(embedding_path, vocab_size, embedding_dim, tokenizer):
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Example usage for SSWE and Emo2vec
sswe_embedding_path = "./dataset/sswe.100d.txt"
emo2vec_embedding_path = "./dataset/emo2vec.100d.txt"

emd_tokenizer = Tokenizer(num_words=VOCAB_SIZE)
emd_tokenizer.fit_on_texts(df_train[TEXT_COLUMN])

sswe_embedding_matrix = load_embedding_matrix(sswe_embedding_path, VOCAB_SIZE, 100, emd_tokenizer)
emo2vec_embedding_matrix = load_embedding_matrix(emo2vec_embedding_path, VOCAB_SIZE, 100, emd_tokenizer)

text_input = Input(shape=(MAX_SEQ_LEN,), dtype=tf.string, name="text_input")

#BERT 
def bert_encode(texts, tokenizer, max_len):
    encodings = tokenizer(texts.tolist(), max_length=max_len, truncation=True, padding=True, return_tensors="tf")
    return encodings["input_ids"], encodings["attention_mask"]

bert_input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="bert_input_ids")
bert_attention_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="bert_attention_mask")
bert_output = bert_model(bert_input_ids, attention_mask=bert_attention_mask)[0]
bert_cls = bert_output[:, 0, :]  # [CLS] token embedding, shape (batch, 768)

# SSWE and Emo2vec input layers (precomputed and provided as input features)
sswe_input = Input(shape=(100,), name="sswe_input")
emo2vec_input = Input(shape=(100,), name="emo2vec_input")

# Concatenate BERT [CLS], SSWE, and Emo2vec embeddings
shared_embedding = Concatenate()([bert_cls, sswe_input, emo2vec_input])  # shape (batch, 968)

#Shared task layer
# Parallel path for SSWE

sswe_cnn = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu")(tf.expand_dims(sswe_input, axis=1))
sswe_bilstm = tf.keras.layers.Bidirectional(GRU(128, return_sequences=True))(sswe_cnn)
sswe_attention = tf.keras.layers.Attention()([sswe_bilstm, sswe_bilstm])
sswe_flat_attention = Flatten()(sswe_attention)

# Parallel path for Combined embedding
combined_cnn = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu")(tf.expand_dims(shared_embedding, axis=1))
combined_bilstm = tf.keras.layers.Bidirectional(GRU(128, return_sequences=True))(combined_cnn)
combined_attention = tf.keras.layers.Attention()([combined_bilstm, combined_bilstm])
combined_flat_attention = Flatten()(combined_attention)

# Parallel path for Emo2vec
emo2vec_cnn = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu")(tf.expand_dims(emo2vec_input, axis=1))
emo2vec_bilstm = tf.keras.layers.Bidirectional(GRU(128, return_sequences=True))(emo2vec_cnn)
emo2vec_attention = tf.keras.layers.Attention()([emo2vec_bilstm, emo2vec_bilstm])
emo2vec_flat_attention = Flatten()(emo2vec_attention)


# Concatenate outputs from all three paths
shared_concat_attention = Concatenate()([sswe_flat_attention, emo2vec_flat_attention, combined_flat_attention])

auxiliary_input = Input(shape=(26,), name="auxiliary_input")

fused_context = Concatenate()([Flatten()(shared_concat_attention), auxiliary_input])

dense_output = Dense(DENSE_UNITS, activation="relu", kernel_regularizer=l2(0.01))(fused_context)
dense_output = Dropout(0.3)(dense_output)
final_output = Dense(1, activation='sigmoid')(dense_output)

model = Model(
    inputs=[bert_input_ids, bert_attention_mask, sswe_input, emo2vec_input, auxiliary_input],
    outputs=final_output,
    name="heal_mind_model"
)

for layer in model.layers:
    if 'tf_bert_model' in layer.name:
        print(layer.name)
        layer.trainable = False

model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=['accuracy', cust_f1])

model.summary()


def prepare_bert_inputs(texts, tokenizer, max_len):
    input_ids, attention_masks = bert_tokenize_texts(texts, tokenizer, max_len)
    return input_ids, attention_masks

def prepare_inputs(data, aux_df):
    # Only process BERT and auxiliary features
    # BERT inputs
    bert_input_ids, bert_attention_masks = prepare_bert_inputs(data[TEXT_COLUMN], bert_tokenizer, MAX_SEQ_LEN)
    aux_features = aux_df.values
    # Prepare SSWE and Emo2vec features
    sswe_features = data['sswe_vec'].values if 'sswe_vec' in data.columns else np.zeros((len(data), 100))
    emo2vec_features = data['emo2vec_vec'].values if 'emo2vec_vec' in data.columns else np.zeros((len(data), 100))
    labels = data[LABEL_COLUMN].values
    return bert_input_ids, bert_attention_masks, sswe_features, emo2vec_features, aux_features, labels

train_inputs = prepare_inputs(df_train, aux_df_train)
val_inputs = prepare_inputs(df_val, aux_df_val)
test_inputs = prepare_inputs(df_test, aux_df_test)

train_bert_ids, train_bert_masks, train_sswe, train_emo2vec, train_aux, train_labels = train_inputs
val_bert_ids, val_bert_masks, val_sswe, val_emo2vec, val_aux, val_labels = val_inputs
test_bert_ids, test_bert_masks, test_sswe, test_emo2vec, test_aux, test_labels = test_inputs

print("Training class distribution:", Counter(train_labels))
print("Validation class distribution:", Counter(val_labels))

# Train
history = model.fit(
    [train_bert_ids, train_bert_masks, train_sswe, train_emo2vec, train_aux],
    train_labels,
    validation_data=([val_bert_ids, val_bert_masks, val_sswe, val_emo2vec, val_aux], val_labels),
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    # callbacks=[tf.keras.callbacks.EarlyStopping(
    #            patience=15,
    #              min_delta=0.01,
    #              baseline=0.9,
    #              mode='auto',
    #              monitor='val_output_accuracy',
    #              restore_best_weights=True,
    #              verbose=1)
)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(f"dataset/{data_name}_history_e{EPOCH}.csv")

#Evaluate
test_loss, test_accuracy, test_f1 = model.evaluate(
    [test_bert_ids, test_bert_masks, test_sswe, test_emo2vec, test_aux],
    test_labels
)
print(f"\n\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}")

test_pred = model.predict([test_bert_ids, test_bert_masks, test_sswe, test_emo2vec, test_aux])
print(test_pred.shape)

pred_labels = [int(np.round(pred.mean())) for pred in test_pred]  

print(f"Epoch: {EPOCH} \nBatch Size: {BATCH_SIZE}")

df_test['predicted_labels'] = pred_labels
df_test.to_csv(f"dataset/{data_name}_predicted.csv")

print(classification_report(test_labels, pred_labels))

pd.DataFrame(classification_report(test_labels, pred_labels, output_dict=True)).transpose().to_csv(f"dataset/{data_name}_report_e{EPOCH}.csv")


# Parallel shared layers for SSWE, Emo2vec, and Combined embedding
def shared_path(x, name_prefix):
    x = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu", name=f"{name_prefix}_cnn")(tf.expand_dims(x, axis=1))
    x = tf.keras.layers.Bidirectional(GRU(128, return_sequences=True), name=f"{name_prefix}_bilstm")(x)
    attn = tf.keras.layers.Attention(name=f"{name_prefix}_attn")([x, x])
    x = Flatten()(attn)
    return x

sswe_shared = shared_path(sswe_input, "sswe")
emo2vec_shared = shared_path(emo2vec_input, "emo2vec")
combined_embedding = Concatenate()([bert_cls, sswe_input, emo2vec_input])
combined_shared = shared_path(combined_embedding, "combined")

# Fuse outputs from three paths
fused_context = Concatenate(name="fused_context")([sswe_shared, emo2vec_shared, combined_shared])

# Task-specific layers start here
# Sentiment task head

# Define the number of sentiment labels (update this value as needed)
num_sentiment_labels = 3  # For three sentiment classes,

sentiment_task_attn = tf.keras.layers.Attention(name="sentiment_task_attn")([fused_context, fused_context])
sentiment_task_dense = Dense(DENSE_UNITS, activation="relu", kernel_regularizer=l2(0.01), name="sentiment_task_dense")(Flatten()(sentiment_task_attn))
sentiment_output = Dense(num_sentiment_labels, activation='softmax', name="sentiment_output")(sentiment_task_dense)

# Emotion task head (parallel)
# Define the number of emotion labels (update this value as needed)
num_emotion_labels = 6  # Dor six emotion classes

emotion_task_attn = tf.keras.layers.Attention(name="emotion_task_attn")([fused_context, fused_context])
emotion_task_dense = Dense(DENSE_UNITS, activation="relu", kernel_regularizer=l2(0.01), name="emotion_task_dense")(Flatten()(emotion_task_attn))
emotion_output = Dense(num_emotion_labels, activation='softmax', name="emotion_output")(emotion_task_dense)

model = Model(
    inputs=[bert_input_ids, bert_attention_mask, sswe_input, emo2vec_input, auxiliary_input],
    outputs=[sentiment_output, emotion_output],
    name="heal_mind_multitask_model"
)

model.compile(
    loss={
        "sentiment_output": "sparse_categorical_crossentropy",
        "emotion_output": "sparse_categorical_crossentropy"
    },
    optimizer='Adam',
    metrics={
        "sentiment_output": ['accuracy', cust_f1],
        "emotion_output": ['accuracy', cust_f1]
    }
)
model.summary()