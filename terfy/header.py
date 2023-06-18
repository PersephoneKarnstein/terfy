import os,re

width = os.get_terminal_size().columns
quote_wide = "\"Go away or I will replace you with a very small shell script\""
quote_narrow = "\"Go away or I will\nreplace you with\na very small\nshell script\""
header_big = """\n\033[38;5;14m████████\033[38;5;212m╗          \033[38;5;14m███████\033[38;5;212m╗          \033[38;5;14m██████\033[38;5;212m╗            \033[38;5;14m███████\033[38;5;212m╗     \n\033[38;5;212m╚══\033[38;5;14m██\033[38;5;212m╔══╝          \033[38;5;14m██\033[38;5;212m╔════╝          \033[38;5;14m██\033[38;5;212m╔══\033[38;5;14m██\033[38;5;212m╗           \033[38;5;14m██\033[38;5;212m╔════╝     \n   \033[38;5;14m██\033[38;5;212m║             \033[38;5;14m█████\033[38;5;212m╗            \033[38;5;14m██████\033[38;5;212m╔╝           \033[38;5;14m█████\033[38;5;212m╗       \n   \033[38;5;14m██\033[38;5;212m║             \033[38;5;14m██\033[38;5;212m╔══╝            \033[38;5;14m██\033[38;5;212m╔══\033[38;5;14m██\033[38;5;212m╗           \033[38;5;14m██\033[38;5;212m╔══╝       \n   \033[38;5;14m██\033[38;5;212m║ \033[38;5;15mRANSPHOBIC, \033[38;5;14m███████\033[38;5;212m╗\033[38;5;15mXTREMELY  \033[38;5;14m██\033[38;5;212m║  \033[38;5;14m██\033[38;5;212m║\033[38;5;15mIDICULOUS  \033[38;5;14m██\033[38;5;212m║  \033[38;5;15mUCKERY  \n   \033[38;5;212m╚═╝             ╚══════╝          ╚═╝  ╚═╝           ╚═╝\033[0m          \n"""
header_small = """\033[38;5;14m___  __  _   __\n\033[38;5;14m |  |_  |_) |_ \n\033[38;5;14m |  |__ | \ |  \n\033[0m"""


def center_text(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    for line in text.split("\n"):
        print((int(width/2 - len(ansi_escape.sub('', line))/2))*" " + line)
def render():
    if width > len(quote_wide):
        center_text(header_big)
        center_text(quote_wide)
    else:
        center_text(header_small)
        center_text("\033[38;5;14mTr\033[38;5;51man\033[38;5;81msp\033[38;5;111mho\033[38;141;51mbi\033[38;5;177mc, \033[38;5;213mEx\033[38;5;219mtr\033[38;5;225mem\033[38;5;231mel\033[38;5;15my R\033[38;5;231mid\033[38;5;225mic\033[38;5;219mul\033[38;5;213mou\033[38;5;177ms \033[38;5;141mFu\033[38;5;111mck\033[38;5;81mer\033[38;5;51my\033[0m\n")
        center_text(quote_narrow)

