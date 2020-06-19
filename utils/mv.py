from tqdm import tqdm
from shutil import move
import sys
import os


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    print(f"move files in {src} to {dst}, press any enter to confirm")
    input()
    for file_ in tqdm(os.listdir(src)):
        move(os.path.join(src, file_), dst)
