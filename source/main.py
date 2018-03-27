# -*- coding: UTF-8 -*-

from name_disambiguation_local import *
import re
import os

if __name__ == '__main__':

    author_names_to_process = []
    dirct = '../data'
    for root,dirs,files in os.walk(dirct):
        for file in files:
            file = str(file)
            if(file.find('author_id_set_')!=-1):
                name = file[14:]

                if(name not in author_names_to_process):
                    author_names_to_process.append(name)

    #print (author_names_to_process)
    COUNT = 0
    for author_name in author_names_to_process:
        name_disambiguation_local(author_name, COUNT)
        print (author_name," is finished")

        COUNT += 1    