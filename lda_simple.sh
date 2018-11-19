#!/bin/sh
#$ -cwd
#$ -e logs -o logs
#$ -l f_node=1 -l h_rt=0:10:0
#$ -M hirohtht@gmail.com
#$ -m beas
#$ -N lda_simple

mkdir $PWD/data/models/lda/tsubame_simple
date '+%H:%M:%S:start job' >> $PWD/data/models/lda/tsubame_simple/log.txt


# Start python under anaconda env
~/.pyenv/versions/anaconda3-5.3.0/envs/py37/bin/python << %_End_of_Python
# -*- coding: utf-8 -*-
from gensim import corpora, models
import os
import multiprocessing
from time import time
from subprocess import Popen

mainpath = os.path.dirname(os.path.abspath(__file__))
corpuspath = mainpath + '/data/models/lda/corpus/'
savepath = mainpath + '/data/models/lda/tsubame_simple/'
logpath = savepath + 'eventlog.md'
t_start = time()

if not os.path.exists(savepath):
    os.makedirs(savepath)


def writelog(s):
    with open(logpath, 'a') as f:
        f.write('{}:{}'.format(time(), s))


def printer(ldamodel, path):
    with open(path, "w") as f:
        text = ""
        for topic in range(ldamodel.num_topics):
            text += "# topic:{}\n".format(topic)
            for word, prob in ldamodel.show_topic(topic):
                text += "{},".format(word)
            text += "\n"
        f.write(text)


if __name__ == '__main__':
    writelog('load corpus')
    dictionary = corpora.Dictionary.load(corpuspath + "filtered.dict")
    corpus = corpora.MmCorpus(corpuspath + "corpus.mm")
    doc_count, word_count = corpus.num_docs, len(dictionary)

    # loggerの起動(single)
    cmd = "~/.pyenv/versions/anaconda3-5.3.0/envs/py37/bin/python cpu_logger.py {}cpu_log_single.md".format(savepath)
    proc = Popen(cmd, shell=True)

    writelog('start lda(single)')
    # start lda
    time_lda = time()
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, passes=20)
    time_lda = time() - time_lda
    lda.save(savepath + 'lda.model')  # 保存

    # make md
    printer(lda, savepath + 'lda.md')

    writelog('finish lda(single)')

    # loggerの終了
    proc.terminate()
    # loggerの起動(multi)
    cmd = "~/.pyenv/versions/anaconda3-5.3.0/envs/py37/bin/python cpu_logger.py {}cpu_log_multi.md".format(savepath)
    proc = Popen(cmd, shell=True)

    writelog('start lda(multi)')
    # start lda using multicore
    cores = multiprocessing.cpu_count() - 1
    time_ldamulti = time()
    lda = models.LdaMulticore(corpus=corpus, id2word=dictionary,
                              num_topics=100, passes=20, workers=cores)
    time_ldamulti = time() - time_ldamulti
    lda.save(savepath + 'lda_{}core.model'.format(cores))

    # make md
    printer(lda, savepath + 'lda_{}core.md'.format(cores))

    writelog('finish lda(multi)')

    time_report = '''
# lda
|||
|:--|:--|
|num_doc|{}|
|num_word|{}|
|single|{} s|
|{}core|{} s|
'''.format(doc_count, word_count, time_lda, cores, time_ldamulti)
    with open(savepath + 'time.md', "w") as file:
        file.write(time_report)

    # loggerの終了
    proc.terminate()


%_End_of_Python

date '+%H:%M:%S:finish job' >> $PWD/data/models/lda/tsbame_simple/log.txt
