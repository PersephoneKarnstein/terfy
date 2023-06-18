from deepmultilingualpunctuation import PunctuationModel
from alive_progress import alive_bar
import os,re,glob

model = PunctuationModel()

def clean_text():
    path = os.getcwd()
    files = glob.glob(path + '/../infowars/*.txt')
    data = ""
    timestamp = re.compile(r'^\s*\[\d{1,2}(:\d{2})?:\d{2}\.\d{3} --> \d{1,2}(:\d{2})?:\d{2}\.\d{3}\]\s+')
    with alive_bar(len(files)) as bar:
        for f in files:
            result = ""
            for line in open(f).readlines():
                result += " "+timestamp.sub('', line).strip() #remove the timestamps
            result = model.restore_punctuation(result)
            data += result
            bar()
    return data

def save_to_corpus(text):
    path = os.getcwd()
    path += "/training-texts/alexjones.txt"
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        f.write(text)

data = clean_text()
save_to_corpus(data)