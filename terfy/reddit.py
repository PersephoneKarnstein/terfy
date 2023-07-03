import os, json,praw,warnings,re
# from alive_progress import alive_bar
from rich.progress import Progress
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()

# FORMAT = "%(message)s"
# logging.basicConfig(
#     level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[PunctuationModel])]
# )

# log = logging.getLogger("rich")
# log.setLevel(logging.INFO)

def read_secrets() -> dict:
    path = os.getcwd()#"/".join(os.getcwd().split("/")[:-1])
    filename = os.path.join(path, "secrets.json")
    try:
        with open(filename, mode='r') as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return {}

def main():
    console.print(Panel("Reading Reddit...".rjust(int(os.get_terminal_size().columns/2))))
    path = os.getcwd()
    secrets = read_secrets()['PRAW']
    reddit = praw.Reddit(
        client_id=secrets["client_id"],
        client_secret=secrets["client_secret"],
        password=secrets["password"],
        user_agent=secrets["user_agent"],
        username=secrets["username"],
    )

    trans = re.compile(r'(trans ((man)|(woman)|(people)|(ideology)|(men)|(women)|(child)|(children)))|(transgender)|(transsexual)|(tranny)|(transvestite)')

    hatesubs = ["conspiracy", "conspiracy_commons", "conservative", "JordanPeterson", "benshapiro", "stevencrowder", "globeskepticism", "DarkEnlightenment"]
    hotlimit = 500
    min_count = 1
    with Progress() as progress:
        with open(path + "/pdf-texts/reddit-full.txt", "w+") as f:
            with open(path + "/training-texts/reddit.txt", "w+") as g:
                for i, subredditname in enumerate(hatesubs):
                    taskname = "task" + str(i)
                    globals()[taskname] = progress.add_task(f"[sky_blue1]Reading r/{subredditname}...", total=hotlimit)
                for i, subredditname in enumerate(hatesubs):
                    taskname = "task" + str(i)
                    # print(f"\033[1;38;5;15m[INFO]\033[0m Now scraping r/{subredditname}...")
                    subreddit = reddit.subreddit(subredditname)
                    for submission in subreddit.hot(limit=hotlimit):
                        searchme = " ".join([submission.title,submission.selftext])
                        mentioned_trans = len(re.findall(trans, searchme))
                        
                        f.write(submission.title) #always write to reddit-full; only write to reddit.txt if it talks about trans ppl
                        f.write(submission.selftext)
                        if mentioned_trans > min_count:
                            g.write(submission.title)
                            g.write(submission.selftext)

                        submission.comments.replace_more(limit=None)
                        for comment in submission.comments.list():
                            f.write(comment.body if len(comment.body)>30 else "")
                            if mentioned_trans > min_count:
                                g.write(comment.body if len(comment.body)>30 else "")
                                
                        progress.update(globals()[taskname], advance=1)

if __name__ == '__main__':
    main()