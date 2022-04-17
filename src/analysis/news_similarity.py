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
        self.data_preprocessing()

        dictionary = self.create_bag_of_words()
        corpus = self.vectorizing_titles(dictionary)
        most_relevant_news = self.calculate_news_similarity(dictionary, corpus)
        return most_relevant_news

    def data_preprocessing(self) -> pd.DataFrame:
        self.tokenizer()
        # сюда можно добавить дополнительных методов предобработки данных
        # TODO: для улучшения данных стоит удалить стоп словаnews_to_check

    def tokenizer(self) -> pd.DataFrame:
        self.__data["tokens"] = self.__data["title"].apply(self.tokenize_every_line)

    def tokenize_every_line(self, string: str) -> Union[list,str]:
        try:
            tokens = list(tokenize(string, lowercase=True, deacc=True, ))
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
        kw_vector = self.tokenize_one_new(dictionary)
        most_relevant_news = self.compare_news(tf_idf, index, kw_vector)
        return most_relevant_news

    def initialize_similarity_matrix(self, corpus:list, feature_cnt:int) -> Tuple[models.TfidfModel, SparseMatrixSimilarity]:
        tf_idf = models.TfidfModel(corpus)
        index = SparseMatrixSimilarity(tf_idf[corpus], num_features=feature_cnt)
        return tf_idf, index

    # стоит объединить с общим токенизатором
    def tokenize_one_new(self, dictionary: dict) -> list:
        kw_vector = dictionary.doc2bow(tokenize(self.__news_to_check))
        return kw_vector

    def compare_news(self, tf_idf: models.TfidfModel, index: SparseMatrixSimilarity, kw_vector: list) -> str:
        self.__data['compare'] = index[tf_idf[kw_vector]]
        sort_similar_texts = self.__data.sort_values(by='compare', ascending=False, ignore_index=True)[
            ['id', 'title', 'compare']]

        # print(sort_similar_texts)
        # print(sort_similar_texts.iloc[1]['title'])
        return sort_similar_texts.iloc[1]['title']
    
    @property
    def data(self):
        return self.__data

    @property
    def new(self):
        return self.__news_to_check

if __name__ == '__main__':
    test_new = 'Криптобиржа Binance выпустит благотворительную карту для украинских беженцев'

    file_name = 'test_file_db.csv'
    data = pd.read_csv(file_name)

    sim = TextSimilarity(data, test_new)
    # сейчас мы каждый раз высчитываем все векторные представления по новой
    most_relevant_news = sim.get_similar_news()

    print('Новость: ' + test_new)
    print('Наиболее похожая на нее: ' + most_relevant_news)
