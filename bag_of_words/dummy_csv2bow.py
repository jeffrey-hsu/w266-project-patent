'''Processes number_of_patents patents claims and organizes them into
a bag-of-words in matrix market format. It also
logs a dictionary for future use if needed.
Patent numbers can be accessed later with the clump
function.'''

# User-set values:
base_file_path = '/home/cameronbell/' # Home directory
number_of_patents = 100000 # Number of patents to be processed
num_topics = 300          # Number of topics in the LSI model


from gensim import utils, corpora, models
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from datetime import datetime
import sys

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

################### FUNCTIONS ###################
def time():
    return str(datetime.now())[5:19]

def rmPunct(dirtyStr):
	splitCleanStr = [ch for ch in dirtyStr if ch not in string.punctuation]
	cleanStr = ''.join(splitCleanStr)
	return(cleanStr)

def prune(doc):
    """This takes a single document and tokenizes the words, removes
    undesirable elements, and prepares it to be loaded into a dictionary.
    """
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
    '''Sorts through the lines, combining them according
    to patent number, and outputs the joined text.'''
	
    # Initialize values
    last_patent_number = 0
    clump_text = ''
    counter = 0
 
    with open(filename, 'r') as f:
    # line = [patent number, claim number, claim text, dependencies,
    # ind_flg, appl_id.]
		
        for i, line in enumerate(f):
            # Ignore the header
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
	            clump_text = ' '.join((clump_text, claim_text))
			
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
	

patent_claims_file = ''.join((base_file_path, 'patent_data/patent_claims_fulltext.csv'))


################ CREATE DICTIONARY ################
print(time(), 'Beginning to load dictionary.')
dictionary = corpora.Dictionary(
    (prune(doc[1])
        for doc in clump(patent_claims_file)), prune_at = 2000000)
print(time(), 'Dictionary loaded; filtering extremes.')
# Remove frequent and infrequent words, and limit tokens to 100,000
dictionary.filter_extremes()
dictionary.compactify()
dictionary.save(''.join((base_file_path, 'patent_data/dummy_dictionary.dict')))
# To load, just use the following code:
# dictionary = corpora.Dictionary.load('.../dummy_dictionary.dict')
print(time(), 'Dictionary saved.')


################## CREATE CORPUS ##################
class MyCorpus(object):
    def __iter__(self):
        for i, num_and_doc in enumerate(clump(patent_claims_file)):
            if i%2000 == 0 and i != 0:
                print('\r%i patents added to corpus. %s' %(i, time()))
	    	sys.stdout.flush()
	    yield dictionary.doc2bow(prune(num_and_doc[1]))

print(time(), 'Building corpus.')
corpus = MyCorpus() 

# Convert the corpus to Market Matrix format and save it.
print(time(), 'Corpus built. Converting to Market Matrix format.')
corpora.MmCorpus.serialize(''.join((base_file_path, 'patent_data/dummy_corpus.mm')), corpus)
print(time(), 'Market Matrix format saved. Process finished.')


############# CONVERT TO TFIDF AND LSI ############
## Load the Market Matrix corpus
mmcorpus = corpora.MmCorpus(''.join((base_file_path, 'patent_data/dummy_corpus.mm')))
## Create the tfidf model object
tfidf = models.TfidfModel(mmcorpus)
## Transform the whole corpus and save it
mmcorpus_tfidf = tfidf[mmcorpus]
corpora.MmCorpus.serialize(''.join((base_file_path, 'patent_data/dummy_corpus_tfidf.mm')), mmcorpus_tfidf)
## Create the lsi model object
dictionary = corpora.Dictionary.load(''.join((base_file_path, 'patent_data/dummy_dictionary.dict')))
lsi = models.LsiModel(mmcorpus, id2word=dictionary, num_topics=num_topics)
num_topics_to_print = 5
if num_topics < 5:
    num_topics_to_print = num_topics
lsi.print_topics(num_topics_to_print)
### Transform the whole corpus and save it
mmcorpus_lsi = lsi[mmcorpus]
corpora.MmCorpus.serialize(''.join((base_file_path, 'patent_data/dummy_corpus_lsi.mm')), mmcorpus_lsi)
