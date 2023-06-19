#adapted from https://github.com/shivam5992/language-modelling/blob/master/model.py and https://towardsdatascience.com/nlp-splitting-text-into-sentences-7bbce222ef17

from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras_preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from alive_progress import alive_bar
from itertools import chain
import keras.utils as ku 
import numpy as np 
import glob,os,nltk

global max_sequence_len

nltk.download('punkt')

tokenizer = Tokenizer()

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

def generate_text(seed_text, next_words, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		# predicted = model.predict_classes(token_list, verbose=0)
		predicted = (model.predict(token_list) > 0.5).astype("int32")		
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text

def get_corpus_data():
    path = os.getcwd()
    files = glob.glob(path + '/training-texts/*.txt')
    data = ""
    for f in files:
        data += open(f).read()
    return data

def save_model(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	# print("Saved model to disk")

def main():
	with alive_bar(title="\033[38;5;14m[INFO]\033[0m Compiling corpus...".ljust(35), stats=False, monitor=False) as bar:
		data = get_corpus_data()

	with alive_bar(title="\033[38;5;14m[INFO]\033[0m Preparing dataset...".ljust(35), stats=False, monitor=False) as bar:
		predictors, label, max_sequence_len, total_words = dataset_preparation(data)

	# with alive_bar(title="\033[38;5;14m[INFO]\033[0m Generating model...".ljust(30), stats=False, monitor=False) as bar:
	print("\033[38;5;14m[INFO]\033[0m Training model...".ljust(35))
	model = create_model(predictors, label, max_sequence_len, total_words)

	with alive_bar(title="\033[38;5;14m[INFO]\033[0m Saving model...".ljust(35), stats=False, monitor=False) as bar:
		save_model(model)

	with alive_bar(title="\033[38;5;14m[INFO]\033[0m Generating text...".ljust(35), stats=False, monitor=False) as bar:
		print(generate_text("we naughty", 3, max_sequence_len))

if __name__ == '__main__':
	main()