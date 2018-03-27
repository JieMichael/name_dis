# -*- coding: UTF-8 -*-

from name_disambiguation_local import *
import re
import os
import time
import logging
import configparser

if __name__ == '__main__':
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    config = configparser.ConfigParser()
    
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
        time_start = time.time()
        name_disambiguation_local(author_name, COUNT)
        #print (author_name," is finished")
        time_end = time.time()
        #print (time_end-time_start)

        logger.info(author_name+" is finished in "+str(time_end-time_start))

        COUNT += 1    