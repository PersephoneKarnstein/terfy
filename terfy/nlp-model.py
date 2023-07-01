#https://stackabuse.com/gpt-style-text-generation-in-python-with-tensorflowkeras/

import os, glob, keras_nlp, nltk.data, random
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

def get_corpus_data():
	path = os.getcwd()
	files = glob.glob(path + '/training-texts/*.txt')
	data = ""
	# files = [files[1]] #delete this line, this is just for testing
	for f in files:
		data += open(f).read()
	return data

texts = get_corpus_data()

# console.print("[pink1]Decimating...")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text_list = tokenizer.tokenize(texts)
text_list = list(filter(None, text_list))

random.shuffle(text_list)

length = len(text_list)
text_train = text_list[:int(0.7*length)]
text_test = text_list[int(0.7*length):int(0.85*length)]
text_valid = text_list[int(0.85*length):]

from tensorflow.keras.layers import TextVectorization

def custom_standardization(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, "\n", " ")
    return sentence

# maxlen = 50
# You can also set calculate the longest sentence in the data - 25 in this case
maxlen = len(max(text_list)) 

vectorize_layer = TextVectorization(
    standardize = custom_standardization,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)

vectorize_layer.adapt(text_list)
vocab = vectorize_layer.get_vocabulary()
vocab_size = len(vocab)

index_lookup = dict(zip(range(len(vocab)), vocab))    
# index_lookup[5]

batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices(text_train)
train_dataset = train_dataset.shuffle(buffer_size=256)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(text_test)
test_dataset = test_dataset.shuffle(buffer_size=256)
test_dataset = test_dataset.batch(batch_size)

valid_dataset = tf.data.Dataset.from_tensor_slices(text_valid)
valid_dataset = valid_dataset.shuffle(buffer_size=256)
valid_dataset = valid_dataset.batch(batch_size)

def preprocess_text(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_dataset = train_dataset.map(preprocess_text)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_text)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = valid_dataset.map(preprocess_text)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)

embed_dim = 128
num_heads = 4

def create_model():
    inputs = keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    x = keras_nlp.layers.TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)(inputs)
    for i in range(4):
        x = keras_nlp.layers.TransformerDecoder(intermediate_dim=embed_dim*2, num_heads=num_heads,                                                             dropout=0.5)(x)
    do = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(do)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", 
        loss='sparse_categorical_crossentropy',
        metrics=[keras_nlp.metrics.Perplexity(), 'accuracy']
    )
    return model

model = create_model()
model.summary()

class TextSampler(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens
    def sample_token(self, logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)
    def on_epoch_end(self, epoch, logs=None):
        decoded_sample = self.start_prompt
        for i in range(self.max_tokens-1):
            tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
            predictions = self.model.predict([tokenized_prompt], verbose=0)
            sample_index = len(decoded_sample.strip().split())-1
            sampled_token = self.sample_token(predictions[0][sample_index])
            sampled_token = index_lookup[sampled_token]
            decoded_sample += " " + sampled_token
        print(f"\nSample text:\n{decoded_sample}...\n")

# First 5 words of a random sentence to be used as a seed
random_sentence = ' '.join(random.choice(text_valid).replace('\n', ' ').split(' ')[:4])
sampler = TextSampler(random_sentence, 30)
reducelr = keras.callbacks.ReduceLROnPlateau(patience=10, monitor='val_loss')

model = create_model()
history = model.fit(train_dataset, 
                    validation_data=valid_dataset,
                    epochs=10, 
                    callbacks=[sampler, reducelr])

def sample_token(logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

def generate_text(prompt, response_length=20):
    decoded_sample = prompt
    for i in range(response_length-1):
        tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
        predictions = model.predict([tokenized_prompt], verbose=0)
        sample_index = len(decoded_sample.strip().split())-1

        sampled_token = sample_token(predictions[0][sample_index])
        sampled_token = index_lookup[sampled_token]
        decoded_sample += " " + sampled_token
    return decoded_sample

def save_model(model,path,filepath="models"):
	# serialize model to JSON
	model_json = model.to_json()
	with open(path+'/'+filepath+'/model.json', "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(path+'/'+filepath+"/model.h5")
	# print("Saved model to disk")

def load_model(filepath="models"):
	path = os.getcwd()
	# with redirect_stdout(open(os.devnull, 'w')):
	json_file = open(path+'/'+filepath+'/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(path+'/'+filepath+"/model.h5")
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print("Loaded model from disk")
	return loaded_model

path = os.getcwd()
save_model(model,path)

generate_text('the truth ultimately is')
