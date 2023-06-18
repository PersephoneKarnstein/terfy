from deepmultilingualpunctuation import PunctuationModel
from alive_progress import alive_bar
import os,re,glob

min_count = 20 #they're fairly long so set it so he has to talk about it a lot for it to count
# if min_count is set to 10 you get a 10.6 MB file; at 20 you get a 2.5 MB file

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
        with alive_bar(len(files)+1, title="\033[38;5;14m[STATUS]\033[0m Reading transcripts...".ljust(35)) as bar:
            bar()
            with open(path, "w+") as g:
                for f in files:
                    result = ""
                    for line in open(f).readlines():
                        result += " "+timestamp.sub('', line).strip() #remove the timestamps
                    #check if the show talks about trans people; only punctuate if it does.
                    mentioned_trans = len(re.findall(trans, result))
                    if mentioned_trans > min_count:
                        date = f.split("/")[-1].split("_")[0]
                        date = date[:4]+"-"+date[4:6]+"-"+date[-2:]
                        print(f"\033[1;38;5;15m[INFO]\033[0m Mentioned trans people {mentioned_trans} times on {date}. Saving.")
                        try:
                            result = model.restore_punctuation(result) #re-insert punctuation
                        except Exception as e:
                            halflen = int(len(result)/2)
                            print(f"\033[38;5;225m[WARNING]\033[0m Transcript too long. Splitting...")
                            model = PunctuationModel()
                            a,b = result[:halflen],result[halflen:]
                            try:
                                foo = model.restore_punctuation(a)
                                foo += model.restore_punctuation(b)
                                result = foo
                            except Exception:
                                print(f"\033[38;5;225m[WARNING]\033[0m Splitting failed. Skipping.")
                                continue
                        g.write(result)
                        j+=1
                    i+=1
                    bar()
    except KeyboardInterrupt:
        print(f"\033[38;5;225m[CANCELLED]\033[0m {i} shows were read; {j} transcribed.")
        return
    print(f"\033[38;5;14m[FINISHED]\033[0m All shows were read and transcribed.")
    

data = clean_text()