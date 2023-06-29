#adapted from https://github.com/shivam5992/language-modelling/blob/master/model.py, https://towardsdatascience.com/nlp-splitting-text-into-sentences-7bbce222ef17, and https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/first_edition/8.1-text-generation-with-lstm.ipynb

# from alive_progress import alive_bar
from rich.console import Console
from rich.table import Table

from itertools import chain
import numpy as np 
import glob,os,nltk,sys,logging,random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'

# from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras_preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.models import Sequential, model_from_json
import keras.utils as ku 

from contextlib import redirect_stdout

nltk.download('punkt', quiet=True)
tokenizer = Tokenizer()

console = Console()

global max_sequence_len

def dataset_preparation(data):

	# basic cleanup
	corpus = [nltk.tokenize.word_tokenize(sentence) for sentence in nltk.sent_tokenize(data)]
	corpus =list(chain.from_iterable(corpus))
	# print(corpus)
	# tokenization	
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1

	# create input sequences using list of tokens
	input_sequences = []
	
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)

	# pad sequences 
	global max_sequence_len
	max_sequence_len = max([len(x) for x in input_sequences])

	# We sample a new sequence every `step` characters
	step = 3

	# This holds our extracted sequences
	sentences = []

	# This holds the targets (the follow-up characters)
	next_chars = []

	for i in range(0, len(data) - max_sequence_len, step):
		sentences.append(data[i: i + max_sequence_len])
		next_chars.append(data[i + max_sequence_len])
	# print('Number of sequences:', len(sentences))

	# List of unique characters in the corpus
	chars = sorted(list(set(data)))
	# print('Unique characters:', len(chars))
	# Dictionary mapping unique characters to their index in `chars`
	char_indices = dict((char, chars.index(char)) for char in chars)

	# Next, one-shot encode the characters into binary arrays.
	# print('Vectorization...')
	x = np.zeros((len(sentences), max_sequence_len, len(chars)), dtype=bool)
	y = np.zeros((len(sentences), len(chars)), dtype=bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			x[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1
	# print('Done.')
	# print(sentences[:50])

	return sentences, x, y, max_sequence_len, total_words, chars, char_indices

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def create_model(sentences, x, y, maxlen, total_words, chars, char_indices):
	global data

	# char_indices = dict((char, chars.index(char)) for char in chars)

	model = Sequential()
	model.add(LSTM(128, input_shape=(maxlen, len(chars))))
	model.add(Dense(len(chars), activation='softmax'))
	optimizer = RMSprop(learning_rate=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
	# model.fit(predictors, label, epochs=100, verbose=1, callbacks=[earlystop])
	for epoch in range(1, 100):
		print('Epoch', epoch)
		# Fit the model for 1 epoch on the available training data
		model.fit(x, y,
			batch_size=256,
			epochs=1, 
			verbose=1,
			callbacks=[earlystop])

		#check how it's doing
		if epoch%10==0:
			# Select a text seed at random
			start_index = random.randint(0, len(data) - max_sequence_len - 1)
			generated_text = data[start_index: start_index + max_sequence_len]
			seed_text = generated_text.strip().replace("\n", " ")
			# print('--- Generating with seed: "' + generated_text + '"')
			
			tabletitle = f"Epoch {epoch}\nSeed: {generated_text}"
			table = Table(title=tabletitle, show_lines=True)
			table.add_column("Temperature", justify="right", style="pink1", no_wrap=True)
			table.add_column("Text", justify="left", style="plum2")

			for temperature in [0.2, 0.5, 1.0]:
				predictions = generate_text(model, seed_text, max_sequence_len, chars, char_indices, temp=temperature, seq_length=200)
				table.add_row(str(temperature), predictions)
			
			console.print(table)

		save_model(model,os.getcwd())
		# print(model.summary())
	return model 

def generate_text(model, seed_text, max_sequence_len, chars, char_indices, temp=0.5, seq_length=400):
	written = seed_text

	# We generate seq_len characters
	for i in range(seq_length):
		sampled = np.zeros((1, max_sequence_len, len(chars)))
		for t, char in enumerate(seed_text):
			sampled[0, t, char_indices[char]] = 1.

		preds = model.predict(sampled, verbose=1)[0]
		next_index = sample(preds, temp)
		next_char = chars[next_index]

		seed_text += next_char
		seed_text = seed_text[1:]

		written += next_char
	return written

def get_corpus_data():
	path = os.getcwd()
	files = glob.glob(path + '/training-texts/*.txt')
	data = ""
	files = files[:4] #delete this line, this is just for testing
	for f in files:
		data += open(f).read()
	data = data.replace("\n", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
	return data

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


def main():
	global data
	path = os.getcwd()
	filepath = "models"
	jsonpath = path+'/'+filepath+'/model.json'
	h5path = path+'/'+filepath+'/model.h5'

	# if not (os.path.isfile(h5path) and os.path.isfile(jsonpath)):
	with console.status("[sky_blue1]Compiling corpus...", spinner="bouncingBar", spinner_style="pink1") as status:
		data = get_corpus_data()

	with console.status("[sky_blue1]Preparing dataset...", spinner="bouncingBar", spinner_style="pink1") as status:
		sentences, x, y, max_sequence_len, total_words, chars, char_indices = dataset_preparation(data)

	# print(input_sequences[:50])
	console.print("[sky_blue1]Training model...\n[italic](this may take a while)")
	model = create_model(sentences, x, y, max_sequence_len, total_words, chars, char_indices)
	console.print(f"max_sequence_len = {max_sequence_len}")

	with console.status("[sky_blue1]Saving model...", spinner="bouncingBar", spinner_style="pink1") as status:
		save_model(model,path)
	# else: #if the model files exist
	# 	with console.status("[sky_blue1]Loading model...", spinner="bouncingBar", spinner_style="pink1") as status:
	# 		model = load_model()
	# 		max_sequence_len = 29

	# with console.status("[sky_blue1]Generating text...", spinner="bouncingBar", spinner_style="pink1") as status:
		# print(generate_text("the transgender", 3, max_sequence_len, model))

if __name__ == '__main__':
	main()