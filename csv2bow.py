'''
Processes patent claims and organizes them into
a bag-of-words in matrix market format. It also
logs a dictionary for future use if needed.
Patent numbers can be accessed later with the clump
function.
'''
import numpy as np
from gensim import utils, corpora, models
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import string
from datetime import datetime
import sys

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

number_of_patents = 10000 # Number of patents to be processed

################### PROCESS PATENT TEXT DOCS ###################
def time():
    return str(datetime.now())[5:19]


def rmPunct(dirtyStr):
    splitCleanStr = [ch for ch in dirtyStr if ch not in string.punctuation]
    cleanStr = ''.join(splitCleanStr)
    return(cleanStr)


def prune(doc):
    '''
    This takes a single document and tokenizes the words, removes
    undesirable elements, and prepares it to be loaded into a dictionary.
    '''
    # Tokenize the document and make it lowercase
    temp = utils.simple_preprocess(doc.lower())

    # Remove freestanding punctuation and punctuation in words
    temp = [w for w in temp if w not in string.punctuation]
    temp = [rmPunct(w) for w in temp]

    # Remove specific tokens if needed
    #temp = [w for w in temp if w not in set([])]

    # Remove stopwords
    temp = [w for w in temp if w not in stopwords.words('english')]

    # Stem the remaining words
    stemmer = SnowballStemmer('english')
    temp = [stemmer.stem(w) for w in temp]
    return temp


def clump(filename):
    '''
    Sorts through the lines, combining them according
    to patent number, and outputs the joined text.
    '''
    # Initialize values
    last_patent_number = 0
    clump_text = ''
    counter = 0
    
    with open(filename, 'r') as f:
    # line = [patent number, claim number, claim text, dependencies, ind_flg, appl_id]
        for i, line in enumerate(f):
            # Ignore header
            if i == 0:
                pass
            else:
                # Retrieve patent number and text, according to format
                if '"' in line:
                    patent_no = line.split('"')[0].split(',')[0]
                    claim_text = line.split('"')[1]
                else:
                    patent_no = line.split(',')[0]
                    claim_text = line.split(',')[2]
                # Add to the string if it's the same patent as the last line
                if patent_no == last_patent_number:
                    clump_text = ' '.join([clump_text, claim_text])
                
                # Output the old line if a new patent is encountered,
                # and reset the values for patent number and text
                else:
                    if last_patent_number != 0:
                        yield last_patent_number, clump_text
                    last_patent_number = patent_no
                    clump_text = claim_text
                    counter += 1
                if counter % number_of_patents == 0:
                    break
                    
        yield last_patent_number, clump_text # Output the last clump as well


################ CREATE DICTIONARY ################
def create_bow_dict(patent_file, dictionary_path):
    '''
    This function creates a dictionary based on the given patent docs.
    To load dictionary, use the following code:
        dictionary = corpora.Dictionary.load('.../dictionary.dict')
    
    Arg: 
        - patent_file: path to the patent file.
        - dictionary_path: path intended to save the dictionary.
          E.g. '/home/[UserName]/patent_data/dictionary.dict'
    '''
    print(time(), 'Beginning to load dictionary.')
    # Create dictionary with generator that returns 1 doc at a time
    dictionary = corpora.Dictionary((prune(doc[1]) for doc in clump(patent_file)),
                                    prune_at = 2000000)
    print(time(), 'Dictionary loaded. Continue to filter tokens with extreme counts.')
    # Remove frequent and infrequent words, and limit tokens to 100,000
    #dictionary.filter_extremes()
    dictionary.compactify()
    # Save dictionary
    dictionary.save(dictionary_path)
    print(time(), 'Dictionary saved.')


################## CREATE CORPUS ##################
class BoWCorpusBuilder(object):
    def __init__(self, patent_file, dictionary):
        self.patent_file = patent_file
        self.dictionary = dictionary
    
    def __iter__(self):
        for i, num_and_doc in enumerate(clump(self.patent_file)):
            if i%10000 == 0 and i != 0:
                print('\r%i patents added to corpus. %s' %(i, time()))
                sys.stdout.flush()
            yield self.dictionary.doc2bow(prune(num_and_doc[1]))
            

def doc2bow(patent_file, dictionary):
    '''
    Args:
        - patent_file
        - dictionary
    '''
    for i, num_and_doc in enumerate(clump(patent_file)):
        if i%10000 ==0 and i != 0:
            print('%i patents added to corpus. %s' %(i+1, time()))
            sys.stdout.flush()
        if i%number_of_patents == 0 and i != 0:
            break
        yield dictionary.doc2bow(prune(num_and_doc[1]))
            
            
def create_bow_corpus(patent_file, dictionary_path, corpus_path):
    '''
    Args:
        - patent_file: patent file path.
        - dictionary_path: dictionary file path. 
          E.g. '/home/[UserName]/patent_data/dictionary.dict'
        - corpus_path: intended path to save the corpus file. 
          E.g. '/home/[UserName]/patent_data/corpus.mm'
    '''
    # Load dictionary
    dictionary = corpora.Dictionary.load(dictionary_path)
    print(time(), 'Building corpus.')
    corpus = BoWCorpusBuilder(patent_file, dictionary) 
    # Convert the corpus to Market Matrix format and save it.
    print(time(), 'Corpus built. Converting to Market Matrix format.')
    corpora.MmCorpus.serialize(corpus_path, [docbow for docbow in doc2bow(patent_file, dictionary)])
    print(time(), 'Market Matrix format saved. Process finished.')
    

################# CONVERT TO TFIDF #################
def build_tfidf(corpus_path, tfidf_corpus_path):
    # Load the Market Matrix corpus
    corpus = corpora.MmCorpus(corpus_path)
    # Create TF-IDF model object
    mod_tfidf = models.TfidfModel(corpus, normalize=True)
    # Transform the whole corpus and save it
    corpus_tfidf = mod_tfidf[corpus]
    corpora.MmCorpus.serialize(tfidf_corpus_path, corpus_tfidf)

    
def build_lsi(corpus_path, dictionary_path, lsi_corpus_path, num_dim=2):
    # Load the Market Matrix corpus
    corpus = corpora.MmCorpus(corpus_path)
    # Create the lsi model object
    dictionary = corpora.Dictionary.load(dictionary_path)
    mod_lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_dim)
    #lsi.print_topics(2)
    # Transform the whole corpus and save it
    corpus_lsi = mod_lsi[corpus]
    corpora.MmCorpus.serialize(lsi_corpus_path, corpus_lsi)

    
################# COMPUTE DOC SIMILARITY ################# 
def doc_cosine(doc1, doc2):
    '''
    Takes in 2 document vectors and compute the cosine similarity based on the vector values
    Args:
        - doc1: [(token_id, value)]
        - doc2: [(token_id, value)]
    '''
    vec_doc1 = [val for num, val in doc1]
    vec_doc2 = [val for num, val in doc2]
    len_doc1 = np.sqrt(sum(i**2 for i in vec_doc1))
    len_doc2 = np.sqrt(sum(i**2 for i in vec_doc2))
    return np.dot(vec_doc1, vec_doc2) / (len_doc1 * len_doc2)
