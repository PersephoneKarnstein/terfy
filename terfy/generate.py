import os,glob
from . import header, clean_infowars, model, reddit

def main():
    #----------------------------------------------------------------
    #   GENERATE CORPUS
    #----------------------------------------------------------------

    path = os.getcwd()
    width = os.get_terminal_size().columns
    banner = "\033[1;38;5;15m~ GENERATING CORPUS ~\033[0m"
    print("\n\n")
    header.center_text(banner)

    # GENERATE THE INFOWARS FILE
    filename = path + "/training-texts/alexjones.txt"
    print(filename)
    if not os.path.exists(filename):
        clean_infowars.clean_text()
    else:
        print("\033[38;5;14m[INFO]\033[0m Infowars corpus exists.")

    # GENERATE THE REDDIT FILE
    filename = os.path.join(path, "/training-texts/reddit.txt")
    if not os.path.exists(filename):
        reddit.main()
    else:
        print("\033[38;5;14m[INFO]\033[0m Reddit corpus exists.")


    # GENERATE OTHER FILES FOR CORPUS

    print("\033[38;5;14m[SUCCESS]\033[0m Corpus generation complete.")

    #----------------------------------------------------------------
    #  TRAIN MODEL
    #----------------------------------------------------------------


    banner = "\033[1;38;5;15m~ GENERATING MODEL ~\033[0m"
    print("\n\n")
    header.center_text(banner)

    model.main()
