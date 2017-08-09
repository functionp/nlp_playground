# coding: utf-8

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamodel import LdaModel
import math
import numpy
from copy import copy
import logging

NUM_OF_TOPICS = 100
WORD2VEC_MODEL_PATH = "data/best_model.model"
LDA_MODEL_PATH = "data/ldamodel1.model"

'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''

class BncSentences:
    def __init__(self, corpus, n=-1):
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        n = self.n
        ret = []
        for line in open(self.corpus):
            line = line.strip().lower()
            if line.startswith("<s "):
                ret = []
            elif line.strip() == "</s>":
                if n > 0:
                    n -= 1
                if n == 0:
                    break
                yield copy(ret)
            else:
                parts = line.split("\t")
                if len(parts) == 3:
                    word = parts[-1]
                    idx = word.rfind("-")
                    word, pos = word[:idx], word[idx+1:]
                    if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
                        pos = "r"
                    if pos == "j":
                        pos = "a"
                    ret.append(gensim.utils.any2unicode(word + "." + pos))

'''
(a) function load_corpus to read a ciroys from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
NEWLINE = "\n"
NUM_OF_WORDS = 20000
def load_corpus(vocabFile, contextFile):
    id2word = {}
    word2id = {}
    vectors = []

    # Load vocabulary file and construct id-word dictionary
    vocab_file_object = open(vocabFile, 'r')
    for id, line in enumerate(vocab_file_object):
        word = line.rstrip(NEWLINE)
        word2id[word] = id
        id2word[id] = word

    # Load context file and construct list of sparce vectors
    context_file_object = open(contextFile, 'r')
    for id, line in enumerate(context_file_object):
        splited_line = line.rstrip(NEWLINE).split(' ')
        dimension = int(splited_line[0])

        # initialize vector with zero
        vector = [0] * int(NUM_OF_WORDS)
        
        # skip target words with zero frequency
        if splited_line[1]:
            index_element = [(int(e) for e in pair.split(':')) for pair in splited_line[1:]]
            for index, element in index_element:
                vector[index] = element

        vectors.append(vector)

    return id2word, word2id, vectors

'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
is_sparse = lambda vector: isinstance(vector[0], tuple)
def convert_to_sparse(dense_vector):
    if not is_sparse(dense_vector):
        return [(i,v) for i,v in enumerate(dense_vector) if v > 0]
    else:
        return dense_vector

def convert_to_dense(sparse_vector, length):
    # do nothing if the vector is not sparse
    if not is_sparse(sparse_vector): return sparse_vector

    dense_vector = [0] * length
    for t in sparse_vector:
        dense_vector[t[0]] = t[1]
    return dense_vector

def unify_form(vector1, vector2):
    """Get vectors and unify to dense same-dimensional form"""

    length = 0

    if is_sparse(vector1) and is_sparse(vector2):
        # define length of sparse vector as the maximum index
        length = sorted(vector1 + vector2, reverse=True)[0][0] + 1
    elif is_sparse(vector1) and not is_sparse(vector2):
        length = len(vector2)
    elif not is_sparse(vector1) and is_sparse(vector2):
        length = len(vector1)

    # unify all vectors as dense coding
    vector1 = convert_to_dense(vector1, length)
    vector2 = convert_to_dense(vector2, length)

    return vector1, vector2


def cosine_similarity(vector1, vector2):
    vector1, vector2 = unify_form(vector1, vector2)

    get_inner_product = lambda v1, v2: numpy.sum(numpy.array(v1) * numpy.array(v2))
    get_norm = lambda v: math.sqrt(numpy.sum(numpy.array(v) ** 2))
    norm1 = get_norm(vector1)
    norm2 = get_norm(vector2)
    if norm1 == 0 or norm2 == 0: return 0
    
    return float(get_inner_product(vector1, vector2)) / (norm1 * norm2)

'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
    tfIdfVectors = []

    # calculate document frequency:
    # convert element which is greater than 1 into 1, and sum over the rows
    freq_matrix = numpy.array(freqVectors)
    df = numpy.minimum(freq_matrix, numpy.ones(freq_matrix.shape)).sum(axis=0)

    def get_tf_idf(i,j):
        if freqVectors[j][i] > 0:
            return (1+ math.log(freqVectors[j][i], 2)) * (1 + math.log(NUM_OF_WORDS / float(df[i] + 1), 2)) # is it possible for document frequency to be zero?
        else:
            return 0

    tfIdfVectors = [[get_tf_idf(i,j) for i in range(NUM_OF_WORDS)] for j in range(NUM_OF_WORDS)]

    return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling, model_name="data/model1.model", n = -1):

    bnc_sentences = BncSentences(corpus, n)

    # the more worker thread, the faster training becomes
    num_of_worker_thread = 2
    trained_model = Word2Vec(bnc_sentences, size=100, window=5, alpha=learningRate, sample=downsampleRate, negative=negSampling, workers=num_of_worker_thread)

    print "Saving Model to {0} ..".format(model_name)
    trained_model.save(model_name)

    return trained_model

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping (optional) mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping, model_name=LDA_MODEL_PATH):

    # Convert to sparse vectors
    sparse_vectors = [convert_to_sparse(vector) for vector in vectors]

    trained_model = LdaModel(sparse_vectors, num_topics=NUM_OF_TOPICS, id2word=wordMapping, update_every=0)

    print "Saving Model to {0} ..".format(model_name)
    trained_model.save(model_name)

    return trained_model

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID, wordMapping):
    return [wordMapping[pair[0]] for pair in ldaModel.get_topic_terms(topicID, topn=15)]

if __name__ == '__main__':
    import sys

    # python question1.py a data/vocabulary.txt data/word_contexts.txt
    
    part = sys.argv[1].lower()
    
    # these are indices for house, home and time in the data. Don't change.
    house_noun = 80
    home_noun = 143
    time_noun = 12
    
    # this can give you an indication whether part a (loading a corpus) works.
    # not guaranteed that everything works.
    if part == "a":
        print("(a): load corpus")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        try:
            if not id2word:
                print("\tError: id2word is None or empty")
                exit()
            if not word2id:
                print("\tError: id2word is None or empty")
                exit()
            if not vectors:
                print("\tError: id2word is None or empty")
                exit()
            print("\tPass: load corpus from file")
        except Exception as e:
            print("\tError: could not load corpus from disk")
            print(e)
        
        try:
            if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
                print("\tError: id2word fails to retrive correct words for ids")
            else:
                print("\tPass: id2word")
        except Exception:
            print("\tError: Exception in id2word")
            print(e)
        
        try:
            if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
                print("\tError: word2id fails to retrive correct ids for words")
            else:
                print("\tPass: word2id")
        except Exception:
            print("\tError: Exception in word2id")
            print(e)
    
    # this can give you an indication whether part b (cosine similarity) works.
    # these are very simple dummy vectors, no guarantee it works for our actual vectors.
    if part == "b":
        #import numpy
        print("(b): cosine similarity")
        try:
            cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
            if not numpy.isclose(0.5, cos):
                print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
            else:
                print("\tPass: sparse vector similarity")
        except Exception:
            print("\tError: failed for sparse vector")
        try:
            cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
            if not numpy.isclose(0.5, cos):
                print("\tError: full expected similarity is 0.5, was {0}".format(cos))
            else:
                print("\tPass: full vector similarity")
        except Exception:
            print("\tError: failed for full vector")
        try:
            cos = cosine_similarity([(0,1), (2,1), (4,2)], [1, 2, 0, 0, 1])
            if not numpy.isclose(0.5, cos):
                print("\tError: full & sparse expected similarity is 0.5, was {0}".format(cos))
            else:
                print("\tPass: full & sparse vector similarity")
        except Exception:
            print("\tError: failed for full & sparse vector")

    # you may complete this part to get answers for part c (similarity in frequency space)
    if part == "c":
        print("(c) similarity of house, home and time in frequency space")
        
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        house_vector = vectors[word2id['house.n']]
        home_vector = vectors[word2id['home.n']]
        time_vector = vectors[word2id['time.n']]
        
        print "Similarity between house.n and home.n is {0}".format(cosine_similarity(house_vector, home_vector))
        print "Similarity between house.n and time.n is {0}".format(cosine_similarity(house_vector, time_vector))
        print "Similarity between home.n and time.n is {0}".format(cosine_similarity(home_vector, time_vector))
    
    # this gives you an indication whether your conversion into tf-idf space works.
    # this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
    if part == "d":
        print("(d) converting to tf-idf space")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        tfIdfSpace = tf_idf(vectors)
        try:
            if not len(vectors) == len(tfIdfSpace):
                print("\tError: tf-idf space does not correspond to original vector space")
            else:
                print("\tPass: converted to tf-idf space")
        except Exception as e:
            print("\tError: could not convert to tf-idf space")
            print(e)
    
    # you may complete this part to get answers for part e (similarity in tf-idf space)
    if part == "e":
        print("(e) similarity of house, home and time in tf-idf space")
        
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        tf_idf_vectors = tf_idf(vectors)
        house_vector = tf_idf_vectors[word2id['house.n']]
        home_vector = tf_idf_vectors[word2id['home.n']]
        time_vector = tf_idf_vectors[word2id['time.n']]
        
        print "Similarity between house.n and home.n is {0}".format(cosine_similarity(house_vector, home_vector))
        print "Similarity between house.n and time.n is {0}".format(cosine_similarity(house_vector, time_vector))
        print "Similarity between home.n and time.n is {0}".format(cosine_similarity(home_vector, time_vector))
    
    # you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
    if part == "f1":
        print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")
        models = []
        hyper_parameters = [(0.025, 1e-5, 5),
                            (0.050, 1e-5, 5),
                            (0.100, 1e-5, 5),
                            (0.050, 0, 5),
                            (0.050, 1e-5, 5),
                            (0.050, 1e-2, 5),
                            (0.050, 1e-5, 0),
                            (0.050, 1e-5, 5),
                            (0.050, 1e-5, 20),]

        for i, hyper_parameter in enumerate(hyper_parameters):
            model_name = "data/model" + str(i) + ".model"
            model = word2vec("data/bnc.vert", learningRate=hyper_parameter[0], downsampleRate=hyper_parameter[1], negSampling=hyper_parameter[2], model_name=model_name, n=1000000)
            models.append(model)
            
        # log accuracy
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        for i, model in enumerate(models):
            print "# # # accuracy for model{0} (learningRate={1}, downsampleRate={2}, negSampling={3}):".format(i, hyper_parameters[i][0], hyper_parameters[i][1], hyper_parameters[i][2])
            model.accuracy("data/accuracy_test.txt")
        
    # you may complete this part for the second part of f (training and saving the actual word2vec model)
    if part == "f2":
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("(f2) word2vec, building full model with best parameters. May take a while.")
        model = word2vec("data/bnc.vert", learningRate=0.1, downsampleRate=1e-2, negSampling=10, model_name=WORD2VEC_MODEL_PATH)
        model.accuracy("data/accuracy_test.txt")
    
    # you may complete this part to get answers for part g (similarity in your word2vec model)
    if part == "g":
        print("(g): word2vec based similarity")
        best_model_name = WORD2VEC_MODEL_PATH
        model = Word2Vec.load(best_model_name)

        print "Similarity between house.n and home.n is {0}".format(model.similarity("house.n", "home.n"))
        print "Similarity between house.n and time.n is {0}".format(model.similarity("house.n", "time.n"))
        print "Similarity between home.n and time.n is {0}".format(model.similarity("home.n", "time.n"))
    
    # you may complete this for part h (training and saving the LDA model)
    if part == "h":
        print("(h) LDA model")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])

        model = lda(vectors, id2word, LDA_MODEL_PATH)

    # you may complete this part to get answers for part i (similarity in your LDA model)
    if part == "i":
        print("(i): lda-based similarity")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        
        model = LdaModel.load(LDA_MODEL_PATH)

        house_vector = model[convert_to_sparse(vectors[word2id['house.n']])]
        home_vector = model[convert_to_sparse(vectors[word2id['home.n']])]
        time_vector = model[convert_to_sparse(vectors[word2id['time.n']])]

        print house_vector
        print home_vector
        print time_vector

        print "Similarity between house.n and home.n is {0}".format(cosine_similarity(house_vector, home_vector))
        print "Similarity between house.n and time.n is {0}".format(cosine_similarity(house_vector, time_vector))
        print "Similarity between home.n and time.n is {0}".format(cosine_similarity(home_vector, time_vector))

    # you may complete this part to get answers for part j (topic words in your LDA model)
    if part == "j":
        print("(j) get topics from LDA model")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])

        model = LdaModel.load(LDA_MODEL_PATH)

        house_vector = model[convert_to_sparse(vectors[word2id['house.n']])]
        home_vector = model[convert_to_sparse(vectors[word2id['home.n']])]
        time_vector = model[convert_to_sparse(vectors[word2id['time.n']])]

        house_vector.sort(key=lambda x:x[1], reverse=True)
        home_vector.sort(key=lambda x:x[1], reverse=True)
        time_vector.sort(key=lambda x:x[1], reverse=True)

        print "top3 topics for the document 'house'"
        for i in range(3):
            print get_topic_words(model, house_vector[i][0], id2word)

        print "top3 topics for the document 'home'"
        for i in range(3):
            print get_topic_words(model, home_vector[i][0], id2word)

        print "top3 topics for the document 'time'"
        for i in range(3):
            print get_topic_words(model, time_vector[i][0], id2word)
