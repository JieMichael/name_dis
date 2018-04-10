#! python3
# -*- coding=utf-8 -*-

import urllib.request
import urllib.parse
from time import sleep
import csv
import _pickle as cpickle

def readcsv(filename):
    csvfile = open(filename,'r')
    reader = csv.reader(csvfile)
    ret = []
    for line in reader:
        ret.append(line)
    csvfile.close()
    return ret 

def writecsv(writed,filename):
    csvfile = open
    csvfile = open(filename, 'w')
    writer = csv.writer(csvfile,lineterminator='\n')
    writer.writerow(('paper_id','mark'))

    for line in writed:
        writer.writerow(line)


# postdata = urllib.parse.urlencode({'pAction': 'zan', 'imgid': '17112121212121', 'wname': 'wname'}).encode('utf-8')
# for i in range(500):
#req = urllib.request.Request(url='http://202.120.36.29:10080', data=b'00001561,77B3BE94,7906CD45,', method='POST')
def post(name):
    req = urllib.request.Request(url='http://202.120.36.29:10080', data=bytes(name), method='POST')
    res = urllib.request.urlopen(req)
    response = res.read()
    clusters = cpickle.loads(response)

    writed = []
    cnt = 0
    for i in clusters:
        cluster = clusters[i]
        cnt +=1
        papers = cluster.papers
        for paper in papers:
            writed.append((paper.paper_id,cnt))
    writecsv(writed,name+".csv")
    sleep(0.5)

if __name__ == '__main__':
    post("zhuo liu")