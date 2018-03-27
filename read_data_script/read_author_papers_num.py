import time
import sys
import pymysql as MySQLdb
import os
import logging
import configparser



def get_paper_affiliations_num_by_author_name(author_name):
    #select stuname as '姓名',classname as '班级' from student inner join c lass on student.stuid=class.stuid
    #select stuname as '姓名',classname as '班级'
    #from student,class
    #where student.stuid=class.stuid
    quest_paper_by_author_name = 'SELECT COUNT(PaperID) FROM PaperAuthorAffiliations AS P INNER JOIN ' \
                                 '(SELECT AuthorID FROM Authors WHERE AuthorName ="%s") AS A ' \
                                 'ON P.AuthorID = A.AuthorID'
    cursor.execute(quest_paper_by_author_name % author_name)
    paper_affiliations_num = cursor.fetchall()
    return paper_affiliations_num

conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205',
                       charset="utf8")
cursor = conn.cursor()


logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log_read_papers_num.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

author_names_to_process = []
max_count = 100000000
COUNT = 0
last_count = 0
for line in open('../AuthorNameIn985.txt',encoding = 'utf-8'):
        COUNT += 1
        if COUNT % 10000000 == 0:
            print (time.now(), COUNT)
            sys.stdout.flush()

        if last_count < COUNT <= max_count:
            author_names_to_process.append(line.replace('\n', ''))
        elif COUNT > max_count:
            break
cnt = 0
f = open('author_paper_num.txt','a')
for author_name in author_names_to_process:
    cnt+=1
    if(cnt %1000==0):
        logger.info("now recording on:"+str(cnt))
    
    paper_affiliations_count = get_paper_affiliations_num_by_author_name(author_name)
    #print (author_name,paper_affiliations_count[0][0])
    f.write(author_name+'\t'+str(paper_affiliations_count[0][0])+"\r")
f.close()