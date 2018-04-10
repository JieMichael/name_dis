import time
import sys
import pymysql as MySQLdb
import _pickle as cpickle
import os
import logging
import configparser
import spacy 
import numpy as np














class Paper:
    def __init__(self, paper_id, title, author_id,nlp,stopwords):
        self.paper_id = paper_id
        # self.title = title
        # self.year = year
        # self.venue_id = venue_id
        # self.affiliation_id = affiliation_id
        # self.coauthors = coauthors
        self.author_id = author_id
        
        title_nlp = nlp(title)

        title_vector_sum = np.zeros(title_nlp[0].vector.shape)
        word_count = 0
        self.title_vector = np.zeros(title_nlp[0].vector.shape)

        for word in title_nlp:
            if str(word) not in stopwords and len(str(word)) > 1:
                title_vector_sum += word.vector
                word_count += 1
        if word_count != 0:
            self.title_vector = title_vector_sum / word_count
        

class Cluster:
    def __init__(self, paper, paper_idx, affiliation_id, year):
        # paper = papers[idx]
        self.author_id = None
        self.papers = list()
        self.papers.append(paper)
        self.paper_idx_list = list()
        self.paper_idx_list.append(paper_idx)
        self.cluster_id = paper_idx
        self.affiliations = set()
        self.affiliations.add(affiliation_id)
        self.year_2_affiliations = dict()
        if affiliation_id is not None and year is not None:
            self.year_2_affiliations[year] = set()
            self.year_2_affiliations[year].add(affiliation_id)

        self.link_type_2_ngbrs = dict()
        self.ngbrs = set()

    def unit(self, other, paper_idx_2_cluster_id):
        for paper_idx in other.paper_idx_list:
            paper_idx_2_cluster_id[paper_idx] = self.cluster_id
        self.papers.extend(other.papers)
        self.paper_idx_list.extend(other.paper_idx_list)
        self.affiliations |= other.affiliations

        for k, v in other.year_2_affiliations.items():
            if k in self.year_2_affiliations.keys():
                self.year_2_affiliations[k] |= v
            else:
                self.year_2_affiliations[k] = v

    def has_no_conflict(self, other, paper_final_edges, strict_mode):
        connected_edges = 0
        for paper_idx in other.paper_idx_list:
            connected_edges += len(np.nonzero(paper_final_edges[paper_idx, self.paper_idx_list])[0])

        if strict_mode and float(connected_edges) < 0.01 * (len(self.papers) * len(other.papers)):
            return False

        if len(self.affiliations | other.affiliations) > 20:
            return False

        for k, v in self.year_2_affiliations.items():
            if k in other.year_2_affiliations.keys():
                if len(v | other.year_2_affiliations[k]) > 3:
                    return False

        return True


def get_paper_affiliations_by_author_name(author_name,cursor):
    #select stuname as '姓名',classname as '班级' from student inner join c lass on student.stuid=class.stuid
    #select stuname as '姓名',classname as '班级'
    #from student,class
    #where student.stuid=class.stuid
    quest_paper_by_author_name = 'SELECT PaperID,AffiliationID,A.AuthorID FROM PaperAuthorAffiliations AS P INNER JOIN ' \
                                 '(SELECT AuthorID FROM Authors WHERE AuthorName ="%s") AS A ' \
                                 'ON P.AuthorID = A.AuthorID'
    cursor.execute(quest_paper_by_author_name % author_name)
    paper_affiliations = cursor.fetchall()
    return paper_affiliations


def get_coauthors_by_paper_id(paper_id,cursor):
    quest_author_by_paper = 'SELECT AuthorID FROM PaperAuthorAffiliations WHERE PaperID = "%s"'
    cursor.execute(quest_author_by_paper % paper_id)
    author_ids = cursor.fetchall()
    if len(author_ids) > 20:
        return None

    quest_author_by_paper = 'SELECT AuthorName FROM Authors INNER JOIN ' \
                            '(SELECT AuthorID FROM PaperAuthorAffiliations WHERE PaperID = "%s") AS TB ' \
                            'ON Authors.AuthorID = TB.AuthorID'
    cursor.execute(quest_author_by_paper % paper_id)
    authors = cursor.fetchall()
    return authors


def get_title_venue_year_by_paper_id(paper_id,cursor):
    quest_info_by_paper = 'SELECT NormalizedPaperTitle, ConferenceSeriesIDMappedToVenueName, ' \
                          'JournalIDMappedToVenueName, PaperPublishYear FROM Papers WHERE PaperID = "%s"'
    cursor.execute(quest_info_by_paper % paper_id)
    rs = cursor.fetchall()
    return rs
def add_in_inverted_indices(inverted_indices, paper_idx, feature_uni_id):
    if feature_uni_id not in inverted_indices:
        inverted_indices[feature_uni_id] = list()
    inverted_indices[feature_uni_id].append(paper_idx)# papers about this unit



def analyze_papers_and_init_clusters(author_name, COUNT,stopwords,nlp,cursor):
    paper_affiliations = get_paper_affiliations_by_author_name(author_name,cursor)
    print("paper_af:"+str(len(paper_affiliations)))
    if len(paper_affiliations) <= 0:
        return None, None, None, None, None
    # elif len(paper_affiliations) > 15000:
    #     f_big = open('./big_name', 'a')
    #     f_big.write(author_name + "\n")
    #     f_big.close()
    #     return None, None, None, None, None

    process_count = 0
    papers = list()
    clusters = dict()
    paper_idx_2_cluster_id = dict()
    inverted_indices = dict()
    author_id_set = set()

    uni_id_generator = 0
    coauthor_2_uni_id = dict()
    affiliation_2_uni_id = dict()
    venue_2_uni_id = dict()

    for paper_affiliation in paper_affiliations:
        paper_id = paper_affiliation[0]
        original_author_id = paper_affiliation[2]
        author_id_set.add(original_author_id)

        # get coauthors
        authors = get_coauthors_by_paper_id(paper_id,cursor)
        if authors is None:
            continue

        paper_idx = process_count

        # coauthors = set()
        for author in authors:
            coauthor_name = author[0]
            if coauthor_name != author_name:
                if coauthor_name not in coauthor_2_uni_id:
                    coauthor_2_uni_id[coauthor_name] = 'a' + str(uni_id_generator)
                    uni_id_generator += 1
                coauthor_uni_id = coauthor_2_uni_id[coauthor_name]
                # coauthors.add(coauthor_uni_id)

                add_in_inverted_indices(inverted_indices, paper_idx, coauthor_uni_id)

        # get affiliation
        affiliation_id = paper_affiliation[1]
        if affiliation_id is not None:
            if affiliation_id not in affiliation_2_uni_id:
                affiliation_2_uni_id[affiliation_id] = 'o' + str(uni_id_generator)
                uni_id_generator += 1
            affiliation_id = affiliation_2_uni_id[affiliation_id]

            add_in_inverted_indices(inverted_indices, paper_idx, affiliation_id)

        # get venue, title and year
        venue_id = None
        title = None
        year = None
        title_venue_year = get_title_venue_year_by_paper_id(paper_id,cursor)
        if len(title_venue_year) != 0:
            # fill in paper_venue_dict
            if title_venue_year[0][1] is not None:
                venue_id = title_venue_year[0][1]
            elif title_venue_year[0][2] is not None:
                venue_id = title_venue_year[0][2]

            if venue_id is not None:
                if venue_id not in venue_2_uni_id:
                    venue_2_uni_id[venue_id] = 'v' + str(uni_id_generator)
                    uni_id_generator += 1
                venue_id = venue_2_uni_id[venue_id]

                add_in_inverted_indices(inverted_indices, paper_idx, venue_id)

            title = title_venue_year[0][0]
            year = title_venue_year[0][3]

        paper_instance = Paper(paper_id, title, original_author_id,nlp,stopwords)
        papers.append(paper_instance)

        # initially each paper is used as a cluster
        new_cluster = Cluster(paper_instance, paper_idx, affiliation_id, year)
        clusters[paper_idx] = new_cluster
        paper_idx_2_cluster_id[paper_idx] = paper_idx
        process_count += 1

    if len(clusters) == 0:
        print ("")
        return None, None, None, None, None

    return papers, clusters, paper_idx_2_cluster_id, inverted_indices, author_id_set



def DownloadData(name,stopwords,nlp,cursor,logger):   
    print("download "+name)
    papers, clusters, paper_idx_2_cluster_id, inverted_indices, author_id_set = analyze_papers_and_init_clusters(name, 100,stopwords,nlp,cursor)
    
    if(papers!= None):
        
        
        
        save_dir = "../data/http/"
        if(os.path.exists(save_dir)==False):
            os.makedirs(save_dir)
        logger.info("make dir:"+save_dir)

        save_dir_now = os.path.join(save_dir,name)
        if(os.path.exists(save_dir_now)==False):
            os.makedirs(save_dir_now)
                
        
            
        cpickle.dump(papers,open(os.path.join(save_dir_now,"papers_%s"%(name)),'wb'))
        cpickle.dump(clusters,open(os.path.join(save_dir_now,"clusters_%s"%(name)),'wb'))
        cpickle.dump(paper_idx_2_cluster_id,open(os.path.join(save_dir_now,"paper_idx_2_cluster_id_%s"%(name)),'wb'))
        cpickle.dump(inverted_indices,open(os.path.join(save_dir_now,"inverted_indices_%s"%(name)),'wb'))
        cpickle.dump(author_id_set,open(os.path.join(save_dir_now,"author_id_set_%s"%(name)),'wb'))

        