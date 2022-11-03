import time
import pandas as pd
from gensim import corpora, models, similarities

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')

#文件加载
def loadData(filename):
    wordArr = [] #存放词语
    wordBag = {} #存放所有出现过的词语

    fr = open(filename, encoding = 'ansi')
    #获取所有出现过的词语，存入字典并计算使用频率
    for line in fr.readlines():
        lineArr = line.strip().split()
        for i in range(1, len(lineArr)):
            if lineArr[i] not in wordBag:
                wordBag[lineArr[i]] = 1
            else:
                wordBag[lineArr[i]] += 1
    print(len(wordBag))  #剩余词语数量:62031

    #删除只出现一次的词语
    for word in list(wordBag.keys()):
        if wordBag[word] < 2:
            del wordBag[word]
    print(len(wordBag)) #剩余词语数量:33003

    #删除无用词
    stopWord = [x[0] for x in wordBag.items() if ('/w' or '/y' or '/u' or '/c') in x[0]]
    for i in range(len(stopWord)):
        del wordBag[stopWord[i]]
    print(len(wordBag)) #剩余词语数量:32966

    fr = open(filename, encoding = 'ansi')
    curArr = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        if lineArr != []:
            for word in lineArr:
                if word in wordBag.keys():
                    curArr.append(word)
        else:
            if curArr != []:
                wordArr.append(curArr)
                curArr = []
    return wordArr


#输出矩阵保存
def saveAsCVS(filename, dataSet):
    filenameStr = filename + '.csv'
    result_csv = pd.DataFrame(dataSet)
    result_csv.to_csv(filenameStr)


#LSI计算
def LSI(texts):
    #用于返回的相似度矩阵
    returnArr = []

    #创建词袋的doc2bow模型
    dictionary = corpora.Dictionary(texts)  # 将所有文本包含的词汇内容转换为字典
    corpus = [dictionary.doc2bow(text) for text in texts]  # 建立每个词袋的doc2bow模型

    # 计算tfidf值，并将词汇的频率替换为tfidf值
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    #构建LSI模型，设置主题数为2
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=4)
    lsi_vector = lsi[corpus_tfidf]

    #建立相似矩阵，用于后续步骤计算相似度
    Similar = similarities.MatrixSimilarity(lsi_vector)

    #对于每一个文本计算与其他文本的相似度
    for i in range(len(texts)):
        #将文本转换为词袋向量
        vec_bow = dictionary.doc2bow(texts[i])
        #将词袋向量转换为tfidf值
        vec_tfidf = tfidf[vec_bow]
        #将tfidf向量转换为lsi向量
        vec_lsi = lsi[vec_tfidf]
        #计算相似度
        sims = Similar[vec_lsi]
        #将该文本与其他文本相似度存入列表
        returnArr.append(sims)

    return returnArr

#main函数
if __name__ == '__main__':
    start = time.time()
    #载入文件
    texts = loadData('renmin.txt')
    print("导入文件完成！")
    #计算所有文本之间的相似度
    result = LSI(texts)
    print("相似度计算完成！正在保存结果......")
    end = time.time()
    #保存
    saveAsCVS('LSI_result',result)
    print("保存结果完成，LSI模型所耗时间为：", end - start, "s")