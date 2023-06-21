import os,re,glob,logging,warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

from deepmultilingualpunctuation import PunctuationModel
from rich.progress import Progress
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[PunctuationModel])]
)

for a in logging.Logger.manager.loggerDict.keys():logging.getLogger(a).disabled = True #HIT WITH BIG STICK TO SHUT THEM UP
logging.getLogger("rich").disabled = False

log = logging.getLogger("rich")
log.setLevel(logging.INFO)



def clean_text():

    min_count = 20 #they're fairly long so set it so he has to talk about it a lot for it to count
    # if min_count is set to 10 you get a 10.6 MB file; at 20 you get a 2.5 MB file
    console.print(Panel("Transcribing Infowars".rjust(int(os.get_terminal_size().columns/2))))
    model = PunctuationModel()

    path = os.getcwd()
    files = glob.glob(path + '/infowars/*.txt')
    path += "/training-texts/alexjones.txt"
    if os.path.exists(path):
        os.remove(path)
    timestamp = re.compile(r'^\s*\[\d{1,2}(:\d{2})?:\d{2}\.\d{3} --> \d{1,2}(:\d{2})?:\d{2}\.\d{3}\]\s+')
    trans = re.compile(r'(trans ((man)|(woman)|(people)|(ideology)|(men)|(women)|(child)|(children)))|(transgender)|(transsexual)|(tranny)|(transvestite)')
    i,j=0,0
    try:
        with Progress() as progress:
            task1 = progress.add_task("[sky_blue1]Reading transcripts...", total=len(files))
            with open(path, "w+") as g:
                for f in files:
                    # progress.console.print(" ")
                    result = ""
                    for line in open(f).readlines():
                        result += " "+timestamp.sub('', line).strip() #remove the timestamps
                    #check if the show talks about trans people; only punctuate if it does.
                    mentioned_trans = len(re.findall(trans, result))
                    if mentioned_trans > min_count:
                        date = f.split("/")[-1].split("_")[0]
                        date = date[:4]+"-"+date[4:6]+"-"+date[-2:]
                        log.info(f"Mentioned trans people {mentioned_trans} times on {date}. Saving.")
                        try:
                            # progress.console.print("Punctuating...")
                            result = model.restore_punctuation(result) #re-insert punctuation
                            # progress.console.print(" ")
                        except Exception as e:
                            halflen = int(len(result)/2)
                            log.warn("Transcript too long. Splitting...")
                            model = PunctuationModel()
                            a,b = result[:halflen],result[halflen:]
                            try:
                                foo = model.restore_punctuation(a)
                                foo += model.restore_punctuation(b)
                                result = foo
                            except Exception:
                                log.warn("Splitting failed. Skipping.")
                                continue
                        g.write(result)
                        j+=1
                    i+=1
                    progress.update(task1, advance=1)
    except KeyboardInterrupt:
        console.print(f"[pink1]Transcription cancelled. {i} shows were read; {j} transcribed.")
        return
    console.print("[pink1]Transcription Complete.")

if __name__ == '__main__':
    clean_text()