# import codecs
# with codecs.open('test1.csv', 'r', 'utf-8-sig') as f:
#     for line in f:
#         print(line)
#         sent = line.split('\t')
#         idx = sent[0]
#         print(int(idx))
#
# with open('test1.csv', 'r') as f:
#     f.read(3)
#     for line in f:
#         print(line)
#         sent = line.split('\t')
#         idx = sent[0]
#         print(int(idx))

import pandas as pd
import collections
a = "a "
words = a.strip().split(" ")
print(words)
