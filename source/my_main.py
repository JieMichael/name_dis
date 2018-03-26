# -*- coding: UTF-8 -*-

from name_disambiguation import *

id_gen = '10000000'
process_id = 'M'
last_count = 0
max_count = 10000

try:
    f = open('./last_count/' + process_id, 'r')
    last_count = int(f.readline())
except:
    last_count = last_count
# last_count = 0

if __name__ == '__main__':

    author_names_to_process = list()
    COUNT = 0
    for line in open('./AuthorNameIn985.txt'):
        COUNT += 1
        if COUNT % 10000000 == 0:
            print datetime.datetime.now(), COUNT
            sys.stdout.flush()

        if last_count < COUNT <= max_count:
            author_names_to_process.append(line.replace('\n', ''))
        elif COUNT > max_count:
            break

    COUNT = last_count + 1
    for author_name in author_names_to_process:
        id_gen = name_disambiguation(author_name, id_gen, process_id, COUNT)
        f = open('./last_count/' + process_id, 'w')
        f.write(str(COUNT))
        COUNT += 1
        f.close()