from gensim import corpora, models
import pickle
import os

mainpath = os.path.dirname(os.path.abspath(__file__))
corpuspath = mainpath + '/data/models/lda/corpus/'
tokenpath = mainpath + '/data/tokens/ieee/'


def tokensgen():
    for dirpath, dirnames, filenames in os.walk(tokenpath):
        for fp in filenames:
            if fp.endswith(".pkl"):
                with open(dirpath + fp, "rb") as f:
                    for paperid, tokens in pickle.load(f):
                        yield tokens


if __name__ == '__main__':
    dictionary = corpora.Dictionary(tokensgen())
    dictionary.save(corpuspath + "origin.dict")
    dictionary.filter_extremes(no_above=0.5, no_below=10)
    dictionary.save(corpuspath + 'filtered.dict')

    def bowgen():
        for tokens in tokensgen():
            yield dictionary.doc2bow(tokens)

    tfidf = models.TfidfModel(bowgen())
    tfidf.save(corpuspath + 'tfidf.model')

    def tfidfgen():
        for bow in bowgen():
            yield tfidf[bow]


    corpora.MmCorpus.serialize(corpuspath + "corpus.mm", tfidfgen())
