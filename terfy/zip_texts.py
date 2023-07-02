import os, glob, nltk.data, zipfile, random

nltk.download('punkt', quiet=True)

def get_corpus_data():
	path = os.getcwd()
	files = glob.glob(path + '/training-texts/*.txt')
	data = ""
	# files = [files[1]] #delete this line, this is just for testing
	for f in files:
		data += open(f).read()
	return data

texts = get_corpus_data()
# length of text is the number of characters in it

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text_list = tokenizer.tokenize(texts)
text_list = list(filter(None, text_list))

print(f'Length of text: {len(texts)} characters; {len(text_list)} sentences.')

random.shuffle(text_list)

length = len(text_list)
text_train = text_list[:int(0.7*length)]
text_test = text_list[int(0.7*length):int(0.85*length)]
text_valid = text_list[int(0.85*length):]

with open("train.txt", 'w') as f:
    f.write("\n\n".join(text_train))

with open("test.txt", 'w') as f:
    f.write("\n\n".join(text_test))
    
with open("valid.txt", 'w') as f:
    f.write("\n\n".join(text_valid))

with zipfile.ZipFile("texts.zip", "w") as f:
	for a in ["train.txt", "test.txt", "valid.txt"]:
		f.write(a)
		os.remove(a)
    
