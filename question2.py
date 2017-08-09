# coding: utf-8

from question1 import *
import json
import re

TEST_DATA_PATH = "data/test.txt"
THESAURUS_PATH = "data/test_thesaurus.txt"
'''
helper class to load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
    thesaurus = {}
    with open(thesaurusFile) as inFile:
        for line in inFile.readlines():
            word, subs = line.strip().split("\t")
            thesaurus[word] = subs.split(" ")
    return thesaurus

def load_test(testFile):
    tests = []
    lines =  open(testFile).readlines()
    for line in lines:
        tests.append(json.loads(line.strip()))

    return tests

def get_context_and_target(test_json, window=5):
    words = test_json['sentence'].split(" ")
    position = int(test_json['target_position'])
    i_begin = max(position - (window - 1),0)
    i_end = position + (window - 1)

    context = words[i_begin:position] + words[position+1: position + window]
    target_word = words[position]

    return target_word, context

'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
    both_sparse = is_sparse(vector1) and is_sparse(vector2)
    vector1, vector2 = unify_form(vector1, vector2)
    result = list(numpy.array(vector1) + numpy.array(vector2))

    if both_sparse:
        return convert_to_sparse(result)
    else:
        return result

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
    both_sparse = is_sparse(vector1) and is_sparse(vector2)
    vector1, vector2 = unify_form(vector1, vector2)
    result =  list(numpy.array(vector1) * numpy.array(vector2))

    if both_sparse:
        return convert_to_sparse(result)
    else:
        return result

'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
    word_vector = convert_to_dense(ldaModel[convert_to_sparse(wordVector)], NUM_OF_TOPICS)
    sum_of_elements = sum(word_vector)
    if sum_of_elements == 0: return 0
    
    return word_vector[topicID] / sum_of_elements

'''
(d) function prob_w_given_z to get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
# for acceleration, target_word_id and topic_term_dict_table is introduced as optional parameter 
def prob_w_given_z(ldaModel, targetWord, topicID, target_word_id=None, topic_term_dict_table=None):
    if topic_term_dict_table == None:
        topic_term_dict = dict(ldaModel.get_topic_terms(topicID, topn=150))
    else:
        topic_term_dict = topic_term_dict_table[topicID]

    if target_word_id == None:
        word2id = {v:k for k,v in ldaModel.id2word.items()} # this process take long
        target_word_id = word2id.get(targetWord,-1)

    return float(topic_term_dict.get(target_word_id,0))


'''
get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
'''
def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType):


    # define the function to convert word to vector according to the model
    if isinstance(model,list):
        def get_vector_from_word(word):
            try:
                return model[word2id[word]]
            except KeyError:
                return None

    elif isinstance(model, gensim.models.word2vec.Word2Vec):
        def get_vector_from_word(word):
            try:
                return model[word]
            except KeyError:
                return None

    elif isinstance(model, gensim.models.ldamodel.LdaModel):
        def get_vector_from_word(word):
            try:
                return model[convert_to_sparse(frequencyVectors[word2id[word]])]
            except KeyError:
                return None

    # (b) use addition to get context sensitive vectors
    if csType == "addition":
        get_context_sensitive_vector = lambda target_word, context_word: addition(get_vector_from_word(target_word), get_vector_from_word(context_word))
        
    # (c) use multiplication to get context sensitive vectors
    elif csType == "multiplication":
        get_context_sensitive_vector = lambda target_word, context_word: multiplication(get_vector_from_word(target_word), get_vector_from_word(context_word))
        
    # (d) use LDA to get context sensitive vectors
    elif csType == "lda":
        # for acceleration, calculate the look-up table for topic_term probability beforehand
        topic_term_dict_table = [dict(ldaModel.get_topic_terms(i, topn=5000)) for i in range(NUM_OF_TOPICS)]
        
        def get_context_sensitive_vector(target_word, context_word):
            context_word_id = word2id.get(context_word,-1)
            targetWordVec = frequencyVectors[word2id[target_word]]
            denominator = sum([prob_z_given_w(model, i, targetWordVec) * prob_w_given_z(model, context_word, i, context_word_id, topic_term_dict_table) for i in range(NUM_OF_TOPICS)])
            if denominator == 0: return [0] * NUM_OF_TOPICS
            return [(prob_z_given_w(model, topicID, targetWordVec) * prob_w_given_z(model, context_word, topicID, context_word_id, topic_term_dict_table)) / float(denominator) for topicID in range(NUM_OF_TOPICS)]

    target_word, context = get_context_and_target(jsonSentence)

    # if the model does not have the target word, return empty value
    if get_vector_from_word(target_word) == None: return ""

    score_thesaurus_list = []
    for thesaurus_word in thesaurus[target_word]:
        thesaurus_vector = get_vector_from_word(thesaurus_word)

        #ignore if the thesaurus word is not in the model
        if thesaurus_vector == None: continue

        score = 0
        for context_word in context:
            #ignore the word if it is not in the vocabulary file or the model
            if word2id.get(context_word, None) == None or get_vector_from_word(context_word) == None: continue
            context_sensitive_vector = get_context_sensitive_vector(target_word, context_word)
            score += cosine_similarity(thesaurus_vector ,context_sensitive_vector)

        score_thesaurus_list.append((score, thesaurus_word))
        #print "{1}: {0}".format(score, thesaurus_word)

    score_thesaurus_list.sort(reverse=True)
    
    return score_thesaurus_list[0][1]

def execute_tests(jsonSentences, thesaurus, word2id, model, frequencyVectors, csType, outputFile):
    output_file = open(outputFile, 'w')

    for jsonSentence in jsonSentences:
        substitute = re.sub("\..","", best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType))
        output_line = "{0} {1} :: {2}\n".format(jsonSentence['target_word'], jsonSentence['id'], substitute)
        print output_line,
        output_file.write(output_line)

    print "Test is done, saved to {0}".format(outputFile)
    output_file.close()

        
if __name__ == "__main__":
    import sys
    
    part = sys.argv[1]
    
    # this can give you an indication whether part a (vector addition and multiplication) works.
    if part == "a":
        print("(a): vector addition and multiplication")
        v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
        try:
            if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
                print("\tError: sparse addition returned wrong result")
            else:
                print("\tPass: sparse addition")
        except Exception as e:
            print("\tError: exception raised in sparse addition")
            print(e)
        try:
            if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
                print("\tError: sparse multiplication returned wrong result")
            else:
                print("\tPass: sparse multiplication")
        except Exception as e:
            print("\tError: exception raised in sparse multiplication")
            print(e)
        try:
            addition(v3,v4)
            print("\tPass: full addition")
        except Exception as e:
            print("\tError: exception raised in full addition")
            print(e)
        try:
            multiplication(v3,v4)
            print("\tPass: full multiplication")
        except Exception as e:
            print("\tError: exception raised in full addition")
            print(e)
    
    # you may complete this to get answers for part b (best substitution words with tf-idf and word2vec, using addition)
    if part == "b":
        print("(b) using addition to calculate best substitution words")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])

        test_json = load_test(TEST_DATA_PATH)

        thesaurus = load_thesaurus(THESAURUS_PATH)

        tfidf_model = tf_idf(vectors)

        w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)

        execute_tests(test_json, thesaurus, word2id, tfidf_model, vectors, "addition", "data/tfidf_addition.txt")
        execute_tests(test_json, thesaurus, word2id, w2v_model, vectors, "addition", "data/word2vec_addition.txt")
    
    # you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
    if part == "c":
        print("(c) using multiplication to calculate best substitution words")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])

        test_json = load_test(TEST_DATA_PATH)

        thesaurus = load_thesaurus(THESAURUS_PATH)

        tfidf_model = tf_idf(vectors)

        w2v_model_name = WORD2VEC_MODEL_PATH
        w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)

        execute_tests(test_json, thesaurus, word2id, tfidf_model, vectors, "multiplication", "data/tfidf_multiplication.txt")
        execute_tests(test_json, thesaurus, word2id, w2v_model, vectors, "multiplication", "data/word2vec_multiplication.txt")
    
    # this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
    if part == "d":
        print("(d): calculating P(Z|w) and P(w|Z)")
        print("\tloading corpus")
        id2word, word2id, vectors=load_corpus(sys.argv[2], sys.argv[3])
        print("\tloading LDA model")

        #ldaModel = gensim.models.ldamodel.LdaModel.load("lda.model")        
        ldaModel = LdaModel.load(LDA_MODEL_PATH)
        
        houseTopic = ldaModel[convert_to_sparse(vectors[word2id["house.n"]])][0][0]
        try:
            if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
                print("\tPass: P(Z|w)")
            else:
                print("\tFail: P(Z|w)")
        except Exception as e:
            print("\tError: exception during P(Z|w)")
            print(e)
        try:
            if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
                print("\tPass: P(w|Z)")
            else:
                print("\tFail: P(w|Z)")
        except Exception as e:
            print("\tError: exception during P(w|Z)")
            print(e)
    
    # you may complete this to get answers for part d2 (best substitution words with LDA)
    if part == "e":
        print("(e): using LDA to calculate best substitution words")

        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])

        test_json = load_test(TEST_DATA_PATH)

        thesaurus = load_thesaurus(THESAURUS_PATH)
        
        ldaModel = LdaModel.load(LDA_MODEL_PATH)
        
        execute_tests(test_json, thesaurus, word2id, ldaModel, vectors, "lda", "data/output_lda.txt")
