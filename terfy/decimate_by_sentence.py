import nltk.data
import re

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("training-texts/alexjones.txt")
data = fp.read()
sentences = tokenizer.tokenize(data)

decimated = []
surrounding = 5

trans = re.compile(r'(trans ((man)|(woman)|(people)|(ideology)|(men)|(women)|(child)|(children)))|(transgender)|(transsexual)|(tranny)|(transvestite)|(penis)|([mM]ichael [oO]bama)')


for i, sentence in enumerate(sentences):
    if trans.search(sentence):
        for j in range(i-surrounding, i+surrounding, 1):
            try: decimated.append(sentences[j])
            except IndexError: pass
            
with open('training-texts/alexjones-decimated.txt', 'w') as f:
    f.write(" ".join(decimated))