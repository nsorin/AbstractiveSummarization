import os


class SentenceReader:

    def __init__(self, dir_list):
        self.dir_list = dir_list

    def __iter__(self):
        for dir in self.dir_list:
            for name in os.listdir(dir):
                for line in open(os.path.join(dir, name), errors="ignore"):
                    if line and line != "\n" and line != "@highlight\n":
                        yield line.split(' ')
                print('Read: ' + dir + " " + name)
            print('Read directory: ' + dir)
