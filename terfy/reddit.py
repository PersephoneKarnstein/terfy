import os, json, praw
from alive_progress import alive_bar

def read_secrets() -> dict:
    path = os.getcwd()#"/".join(os.getcwd().split("/")[:-1])
    filename = os.path.join(path, "secrets.json")
    try:
        with open(filename, mode='r') as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return {}

def main():
    path = os.getcwd()
    secrets = read_secrets()['PRAW']
    reddit = praw.Reddit(
        client_id=secrets["client_id"],
        client_secret=secrets["client_secret"],
        password=secrets["password"],
        user_agent=secrets["user_agent"],
        username=secrets["username"],
    )

    hatesubs = ["conspiracy", "conspiracy_commons", "conservative", "JordanPeterson", "benshapiro", "stevencrowder", "globeskepticism"]
    hotlimit = 200
    with alive_bar(len(hatesubs)*hotlimit, title="\033[38;5;14m[STATUS]\033[0m Reading reddit...".ljust(35)) as bar:
        for subredditname in hatesubs:
            with open(path + "/training-texts/reddit.txt", "w+") as g:
                subreddit = reddit.subreddit(subredditname)
                for submission in subreddit.hot(limit=hotlimit):
                    g.write(submission.title)
                    g.write(submission.selftext)
                    submission.comments.replace_more(limit=None)
                    for comment in submission.comments.list():
                        g.write(comment.body if len(comment.body)>30 else "")
                    bar()

if __name__ == '__main__':
    main()