# This class is used as an iterator by word2vec.py to iterate through the source files.

import os


class SentenceReader:

    def __init__(self, dir_list):
        self.dir_list = dir_list

    def __iter__(self):
        for directory in self.dir_list:
            for name in os.listdir(directory):
                for line in open(os.path.join(directory, name), errors="ignore"):
                    # Ignore blank lines and "@highlight" markers
                    if line and line != "\n" and line != "@highlight\n":
                        yield line.split(' ')
                print('Read: ' + directory + " " + name)
            print('Read directory: ' + directory)
