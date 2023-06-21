import os,glob
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from . import header, clean_infowars, model, reddit

console = Console()

def main():
    #----------------------------------------------------------------
    #   GENERATE CORPUS
    #----------------------------------------------------------------

    path = os.getcwd()
    console.print(Panel(Padding("[bold]"+"☙ GENERATING CORPUS ❧".rjust(int(os.get_terminal_size().columns/2)),1),border_style="pink1"))
    consolr.print("\n\n")

    # GENERATE THE INFOWARS FILE
    filename = path + "/training-texts/alexjones.txt"
    # print(filename)
    if not os.path.exists(filename):
        clean_infowars.clean_text()
    else:
        console.print("[sky_blue1]Infowars corpus exists.")

    # GENERATE THE REDDIT FILE
    filename = path+ "/training-texts/reddit.txt"
    if not os.path.exists(filename):
        reddit.main()
    else:
        console.print("[sky_blue1]Reddit corpus exists.")


    # GENERATE OTHER FILES FOR CORPUS

    console.print("[sky_blue1]Corpus generation complete.")
    consolr.print("\n\n")

    #----------------------------------------------------------------
    #  TRAIN MODEL
    #----------------------------------------------------------------


    console.print(Panel(Padding("[bold]"+"☙ GENERATING MODEL ❧".rjust(int(os.get_terminal_size().columns/2)),1),border_style="pink1"))
    console.print("\n\n")

    model.main()
