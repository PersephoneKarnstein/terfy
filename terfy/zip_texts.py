import os, glob, nltk.data, tarfile, random

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
# text_list = list(filter(None, text_list))
text_list = [a for a in text_list if len(a)>3]

print(f'Length of text: {len(texts)} characters; {len(text_list)} sentences.')

random.shuffle(text_list)

with open("valid.txt", 'w') as f:
    f.write("\n\n".join(text_list))
    
with open("train.txt", 'w') as f:
    with open("pdf-texts/alexjones.txt", 'r') as g:
        data = g.read()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(data)
        f.write("\n\n".join(sentences))

with tarfile.open("texts.tar.gz", "w:gz") as tarhandle:
      for a in ["valid.txt","train.txt"]:
            tarhandle.add(a)
            os.remove(a)
# with zipfile.ZipFile("texts.zip", "w") as f:
# 	for a in ["train.txt", "test.txt", "valid.txt"]:
# 		f.write(a)
# 		os.remove(a)
    
