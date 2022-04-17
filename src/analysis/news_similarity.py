from typing import Union

import pandas as pd
from typing import Tuple
from gensim import corpora, models
from gensim.similarities import SparseMatrixSimilarity

from gensim.utils import tokenize
import pymorphy2


class TextSimilarity():

    def __init__(self, data: pd.DataFrame, news_to_check: str) -> None:
        self.__data = data.copy()
        self.__news_to_check = news_to_check

    def get_similar_news(self):
        self.preprocessing_dataframe()

        dictionary = self.create_bag_of_words()
        corpus = self.vectorizing_titles(dictionary)
        sort_similar_texts = self.calculate_news_similarity(dictionary, corpus)
        most_relevant_news = self.threshold_determination(sort_similar_texts)
        return most_relevant_news


    def preprocessing_dataframe(self):
        self.__data["tokens"] = self.__data["title"].apply(self.row_preprocessing)


    def row_preprocessing(self,row:str) -> str:
        preprocessed_row = self.tokenizer(row)
        # сюда можно добавить дополнительных методов предобработки данных
        # TODO: для улучшения данных стоит удалить стоп словаnews_to_check

        return preprocessed_row


    def tokenizer(self, row: str) -> Union[list,str]:
        try:
            tokens = list(tokenize(row, lowercase=True, deacc=True, ))
            # morph = pymorphy2.MorphAnalyzer()
            # for i,token in enumerate(tokens):
            #     print(token)
            #     print(morph.normal_forms(token)[1])
            #     tokens[i] = morph.normal_forms(token)[1]
            #     print(tokens[i])
            return tokens
        except:
            return ""

    def create_bag_of_words(self) -> dict:
        # мешок слов (в дальнейшем можно будет обновлять каждый час, например)
        dictionary = corpora.Dictionary(self.__data["tokens"])
        # print(dictionary.token2id)
        return dictionary

    def update_bag_of_words(self) -> dict:
        # тут в дальнейшем стоит дописывать новые слова в словарь

        # Возможно это стоит делать каждый раз, когда мы
        # сравниваем новое предложение с текущими
        pass

    def vectorizing_titles(self, dictionary: dict):
        corpus = [dictionary.doc2bow(text) for text in self.__data["tokens"]]
        return corpus

    def calculate_news_similarity(self, dictionary:dict, corpus:list) -> str:
        feature_cnt = len(dictionary.token2id)
        tf_idf, index = self.initialize_similarity_matrix(corpus, feature_cnt)
        kw_vector = self.vectorize_new_to_check(dictionary)
        most_relevant_news = self.compare_news(tf_idf, index, kw_vector)
        return most_relevant_news

    def initialize_similarity_matrix(self, corpus:list, feature_cnt:int) -> Tuple[models.TfidfModel, SparseMatrixSimilarity]:
        tf_idf = models.TfidfModel(corpus)
        index = SparseMatrixSimilarity(tf_idf[corpus], num_features=feature_cnt)
        return tf_idf, index

    def vectorize_new_to_check(self, dictionary: dict) -> list:
        preprocessed_new = self.row_preprocessing(self.__news_to_check)
        kw_vector = dictionary.doc2bow(preprocessed_new)
        return kw_vector

    def compare_news(self, tf_idf: models.TfidfModel, index: SparseMatrixSimilarity, kw_vector: list) -> str:
        self.__data['compare'] = index[tf_idf[kw_vector]]
        sort_similar_texts = self.__data.sort_values(by='compare', ascending=False, ignore_index=True)[
            ['id', 'title', 'compare']]

        # print(sort_similar_texts)
        # print(sort_similar_texts.iloc[1]['title'])
        # return sort_similar_texts.iloc[1]['title']
        return sort_similar_texts

    def threshold_determination(self,sort_similar_texts:pd.DataFrame) -> pd.DataFrame:
        max_num_to_check = 5
        duplicate_threshold = 0.95
        similarity_threshold = 0.11

        num = 0
        while num <= max_num_to_check:
            probability = sort_similar_texts.iloc[num]['compare']

            # добавить проверку по ключевым словам еще ОБЯЗАТЕЛЬНО
            isNotDublicat = probability < duplicate_threshold
            isSimilar = probability > similarity_threshold

            if isNotDublicat and isSimilar:
                print(sort_similar_texts.iloc[num]['compare'])
                # print(sort_similar_texts.iloc[num]['title'])
                return sort_similar_texts.iloc[num]['title']

            num += 1




        return ' '
    
    @property
    def data(self):
        return self.__data

    @property
    def new(self):
        return self.__news_to_check


