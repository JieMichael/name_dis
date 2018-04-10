from read_data_from_db import *
from name_disambiguation_local import *

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log2.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205',
                       charset="utf8")
cursor = conn.cursor()

nlp = spacy.load('en_core_web_md')
logger.info( 'spacy.load finished')

stopwords = set()
for line in open('./stopwords_ace.txt'):
    stopwords.add(line.replace('\n', '').replace('\r', ''))

def HttpDis(name):
    DownloadData(name,stopwords,nlp,cursor,logger)

    save_dir = "../data/http/"
    save_dir_now = os.path.join(save_dir,name)
    print (save_dir_now)
    name_disambiguation_local(name,100, save_dir_now,logger)
    save_dir_result = os.path.join(save_dir_now,"result_cluster_%s"%(name))
    clusters = cpickle.load(open(save_dir_result,"rb"))
    b_clusters = cpickle.dumps(clusters)
    return b_clusters

if __name__ == '__main__':
    result = HttpDis("zhuo liu")