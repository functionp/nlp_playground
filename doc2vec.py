# coding: utf-8

import gensim

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer

import os

def split_to_words(string):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(string)
    words = []
    words = [token.surface for token in tokens if len(token.surface) > 0]

    return words

def learn_doc2vec(root_directory):

    print("Loading File..")

    lines_list = []
    for file_name in os.listdir(root_directory):
        file_path = root_directory + file_name
        file_pointer = open(file_path, "r")

        # 最初の2業はURLがと時間なのでスキップ
        lines = file_pointer.readlines()[2:]
        lines_list.append((lines, file_name))

    print("Loading Documents..")

    tagged_documents_by_sentence = []
    tagged_documents_by_document = []
    for lines, file_name in lines_list:

        words_in_documents = []
        for line in lines:
            words = split_to_words(line)

            if words == []:
                break

            tagged_documents_by_sentence.append(TaggedDocument(words, [line]))
            words_in_documents += words

        # 学習をひとつの文書単位にしたければこのコメントを外す
        tagged_documents_by_document.append(TaggedDocument(words, [file_name]))

    print("Learning doc2vec by Sentence..")
    model_sentence = Doc2Vec(documents=tagged_documents_by_sentence, size=128, window=8, min_count=5, workers=8)
    model_sentence.save("model/doc2vec_sentence.model")

    print("Learning doc2vec by Document..")
    model_document = Doc2Vec(documents=tagged_documents_by_document, size=128, window=8, min_count=5, workers=8)
    model_document.save("model/doc2vec_document.model")

# learn_doc2vec('data/text/kaden-channel/')

def load_doc2vec(model):
    model = Doc2Vec.load(model)
    sims = model.docvecs.most_similar(1)

    vector = model.infer_vector(split_to_words("毎年、ただなんとなく節分を迎えている人もいるのではないだろうか？"))
    most_similar_texts = model.docvecs.most_similar([vector])
    for similar_text in most_similar_texts:
        print(similar_text)

load_doc2vec("model/doc2vec_sentence.model")
