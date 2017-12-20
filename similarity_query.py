'''This file delivers the function similarity_query, which
is described in more detail within the function definition.

To run this function, simply type the following commands:
from similarity_query import similarity_query, show_best
best = similarity_query(3930278)
show_best(best)'''

from gensim import corpora, similarities
import csv2bow
import os

base_path = os.getenv('HOME') + '/patent_data/'
patent_claims_file = base_path + 'patent_claims_fulltext.csv'
dictionary_path = base_path + 'dummy_dictionary.dict'
lsi_corpus_path = base_path + 'dummy_corpus_lsi.mm'
similarity_index_file = base_path + 'lsi_index.index'

# Load dictionary, corpus, and index
dictionary = corpora.Dictionary.load(dictionary_path)
lsi_corpus = corpora.MmCorpus(lsi_corpus_path)
sims_index = similarities.Similarity.load(similarity_index_file)

def patentnum2patent(patent_no):
    '''Returns a tuple (index, text)'''
    for i, clump in enumerate(csv2bow.clump(patent_claims_file, len(lsi_corpus))):
        if clump[0] == patent_no:
            return i, clump[1] # return the index in the corpus and the patent text

def index2patentnum(index):
    for i, clump in enumerate(csv2bow.clump(patent_claims_file, len(lsi_corpus))):
        if index == i:
            return clump[0] # return the patent number

def similarity_query(patent_no, num_best=10):
    '''Uses patent number to retrieve the LSI vector from the loaded
    corpus, then returns the desired amount of similar patents. Returns
    a list of tuples: (patent number, text, similarity score).'''
    patent_index, patent_text = patentnum2patent(str(patent_no))
    vec_lsi = lsi_corpus[patent_index]
    sims_index.num_best = num_best
    best_vecs = sims_index[vec_lsi] # Format: doc_index, similarity (-1,1)
                                    # [(4, 0.8), (2, 0.13), (3, 0.13)]
    best = []
    for vec in best_vecs:
        patentnum = index2patentnum(vec[0])
        best.append((patentnum, patentnum2patent(patentnum)[1], vec[1])) # (Patent number, text, score)
    return best

def show_best(best_patents, category='scores'):
    '''The input for this function is the output of similarity_query -
    a tuple of string, string, float:
    (patent number, patent text, similarity score)'''
    print('%i most similar patents:' % len(best_patents))
    for patent in best_patents:
        if category == 'scores':
            print('Patent %s: %.3f' % (patent[0], patent[2]))
        if category == 'text':
            print(patent[1])
