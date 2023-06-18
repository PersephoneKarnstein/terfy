from deepmultilingualpunctuation import PunctuationModel
from alive_progress import alive_bar
import os,re,glob

model = PunctuationModel()

def clean_text():
    path = os.getcwd()
    files = glob.glob(path + '/../infowars/*.txt')
    path += "/training-texts/alexjones.txt"
    if os.path.exists(path):
        os.remove(path)
    timestamp = re.compile(r'^\s*\[\d{1,2}(:\d{2})?:\d{2}\.\d{3} --> \d{1,2}(:\d{2})?:\d{2}\.\d{3}\]\s+')
    trans = re.compile(r'(trans ((man)|(woman)|(people)|(ideology)|(men)|(women)|(child)|(children)))|(transgender)|(transsexual)|(tranny)|(transvestite)')
    i,j=0,0
    try:
        with alive_bar(len(files)+1, title="\033[38;5;14m[INFO]\033[0m Cleaning transcripts...".ljust(35)) as bar:
            bar()
            with open(path, "w+") as g:
                for f in files:
                    result = ""
                    for line in open(f).readlines():
                        result += " "+timestamp.sub('', line).strip() #remove the timestamps
                    #check if the show talks about trans people; only punctuate if it does.
                    mentioned_trans = len(re.findall(trans, result))
                    if mentioned_trans > 5:
                        print(f"\033[1;38;5;15m[INFO]\033[0m Mentioned trans people {mentioned_trans} times. Saving.")
                        result = model.restore_punctuation(result) #re-insert punctuation
                        g.write(result)
                        j+=1
                    i+=1
                    bar()
    except KeyboardInterrupt:
        print(f"\033[38;5;225m[CANCELLED]\033[0m {i} shows were read; {j} were transcribed.".ljust(35))
    return
    

data = clean_text()