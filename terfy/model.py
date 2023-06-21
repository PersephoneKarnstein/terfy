#adapted from https://github.com/shivam5992/language-modelling/blob/master/model.py and https://towardsdatascience.com/nlp-splitting-text-into-sentences-7bbce222ef17

# from alive_progress import alive_bar
from rich.console import Console

from itertools import chain
import numpy as np 
import glob,os,nltk,sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras_preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_json
import keras.utils as ku 

nltk.download('punkt', quiet=True)
tokenizer = Tokenizer()

console = Console()

global max_sequence_len

def dataset_preparation(data):

	# basic cleanup
	corpus = [nltk.tokenize.word_tokenize(sentence) for sentence in nltk.sent_tokenize(data)]
	corpus =list(chain.from_iterable(corpus))

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
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

	# create predictors and label
	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
	label = ku.to_categorical(label, num_classes=total_words)

	return predictors, label, max_sequence_len, total_words

def create_model(predictors, label, max_sequence_len, total_words):
	
	model = Sequential()
	model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
	model.add(LSTM(150, return_sequences = True))
	# model.add(Dropout(0.2))
	model.add(LSTM(100))
	model.add(Dense(total_words, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
	model.fit(predictors, label, epochs=100, verbose=1, callbacks=[earlystop])
	print(model.summary())
	return model 

def generate_text(seed_text, next_words, max_sequence_len, model):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		# predicted = model.predict_classes(token_list, verbose=0)
		predicted = (model.predict(token_list) > 0.5).astype("int32")		
		output_word = ""
		for word, index in tokenizer.word_index.items():
			console.print("\n\n",word,index)
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text

def get_corpus_data():
	path = os.getcwd()
	files = glob.glob(path + '/training-texts/*.txt')
	data = ""
	files = [files[1]] #delete this line, this is just for testing
	for f in files:
		data += open(f).read()
	return data

def save_model(model,filepath="models"):
	# serialize model to JSON
	model_json = model.to_json()
	with open(path+'/'+filepath+'/model.json', "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(path+'/'+filepath+"/model.h5")
	# print("Saved model to disk")

def load_model(filepath="models"):
	path = os.getcwd()
	with redirect_stdout(open(os.devnull, 'w')):
		json_file = open(path+'/'+filepath+'/model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(path+'/'+filepath+"/model.h5")
	print("Loaded model from disk")
	return loaded_model


def main():
	with console.status("[sky_blue1]Compiling corpus...", spinner="bouncingBar", spinner_style="pink1") as status:
		data = get_corpus_data()

	with console.status("[sky_blue1]Preparing dataset...", spinner="bouncingBar", spinner_style="pink1") as status:
		predictors, label, max_sequence_len, total_words = dataset_preparation(data)

	console.print("[sky_blue1]Training model...\n[italic](this may take a while)")
	model = create_model(predictors, label, max_sequence_len, total_words)

	with console.status("[sky_blue1]Saving model...", spinner="bouncingBar", spinner_style="pink1") as status:
		save_model(model)

	with console.status("[sky_blue1]Generating text...", spinner="bouncingBar", spinner_style="pink1") as status:
		print(generate_text("the transgender", 3, max_sequence_len, model))

if __name__ == '__main__':
	main()