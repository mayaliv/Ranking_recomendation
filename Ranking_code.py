import gensim
import collections
import random
import pandas as pd
import os
import nltk
import numpy as np

# import the data from CSV:
dataset=pd.read_csv('Documents_ranking.csv')



df_english=dataset[dataset['language'].str.contains('english')]

df_water=df_english[df_english['themes'].str.contains('Water')]


data=df_water[['id','text']]
#print(data.head())
print(data.shape)
word_count=[]

for doc in df_water['text']:
    word_count.append(len(doc.split()))

mean_word_count=np.mean(word_count)

# Randomly sample 70% of your dataframe
train_data= data.sample(frac=0.7)

test_data = data.loc[~data.index.isin(train_data.index)]

# Set file names for train and test data
"""test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])"""
train_corpus=[]
test_corpus=[]
for i, line in enumerate(train_data['text']):
        # For training data, add tags
        elem=gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line, deacc=True), [i])
        #elem=nltk.word_tokenize(line)
        #elem=nltk.pos_tag(elem)
        train_corpus.append(elem)

for line2 in test_data['text']:
    elem2=gensim.utils.simple_preprocess(line2, deacc=True)
    test_corpus.append(elem2)

# train_corpus = list(read_corpus(lee_train_file))
# test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
# model = gensim.models.doc2vec.Doc2Vec(alpha=0.025, vector_size=1200, min_count=2, epochs=100)
# model.build_vocab(train_corpus)

model = gensim.models.Doc2Vec(vector_size=mean_word_count, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025,epochs=20)
model.build_vocab(train_corpus)
model.train(train_corpus, epochs=model.iter, total_examples=model.corpus_count)

sim_water_list=model.infer_vector('water')
print(np.max(sim_water_list),np.mean(sim_water_list))

# lda = gensim.models.ldamodel.LdaModel(train_corpus, num_topics=50)
#
# print(lda)

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

print(collections.Counter(ranks))  # Results vary due to random seeding and very small corpus
doc_id=random.choice(range(len(train_corpus)))
print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

# Pick a random document from the test corpus and infer a vector from the model
#doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the most/median/least similar documents from the train corpus
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))