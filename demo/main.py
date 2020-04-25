# /usr/bin/env python
# coding=utf-8
import sys
import random


def process(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')

            fout.write(lineno + "\t" + str(random.randrange(2)) + "\n")


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
