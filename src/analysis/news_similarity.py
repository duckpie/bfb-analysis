


import pandas as pd
import re
from gensim import corpora,models,similarities
from gensim.utils import tokenize
import pymorphy2


morph = pymorphy2.MorphAnalyzer()


class TextSimilarity():

    def __init__(self,file_name:str,news_to_check:str) ->None:
        self.file_name = file_name
        self.news_to_check = news_to_check

    def get_similar_news(self):
        data = self.load_data()
        preprocess_data = self.data_preprocessing(data)
        # print(preprocess_data)

        dictionary = self.create_bag_of_words(data)
        corpus = self.vectorizing_titles(data, dictionary)
        # print(corpus)
        most_relevant_news = self.calculate_news_similarity(data,dictionary,corpus)
        return  most_relevant_news


    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.file_name)
        return data

    def data_preprocessing(self,data:pd.DataFrame) -> pd.DataFrame:
        preprocess_data = self.tokenizer(data)
        # сюда можно добавить дополнительных методов предобработки данных
        # TODO: для улучшения данных стоит удалить стоп словаnews_to_check
        return preprocess_data


    def tokenizer(self,data:pd.DataFrame) -> pd.DataFrame:
        data["tokens"] = data["title"].apply(self.tokenize_every_line)
        return data

    def tokenize_every_line(self,string:str) -> list:
        try:
            tokens = list(tokenize(string, lowercase=True, deacc=True, ))
            # for i,token in enumerate(tokens):
            #     print(token)
            #     print(morph.normal_forms(token)[1])
            #     tokens[i] = morph.normal_forms(token)[1]
            #     print(tokens[i])
            return tokens
        except:
            return ""

    def create_bag_of_words(self,data:pd.DataFrame):
        # мешок слов (в дальнейшем можно будет обновлять каждый час, например)
        dictionary = corpora.Dictionary(data["tokens"])
        # print(dictionary.token2id)
        return dictionary

    def update_bag_of_words(self,data:pd.DataFrame):
        # тут в дальнейшем стоит дописывать новые слова в словарь

        # Возможно это стоит делать каждый раз, когда мы
        # сравниваем новое предложение с текущими
        pass

    def vectorizing_titles(self,data:pd.DataFrame,dictionary:dict):
        corpus = [dictionary.doc2bow(text) for text in data["tokens"]]
        return corpus

    def calculate_news_similarity(self,data,dictionary,corpus):
        feature_cnt = len(dictionary.token2id)
        tf_idf, index = self.initialize_similarity_matrix(corpus,feature_cnt)
        kw_vector = self.tokenize_one_new(dictionary)
        most_relevant_news = self.compare_news(data,tf_idf,index,kw_vector)
        return most_relevant_news


    def initialize_similarity_matrix(self,corpus,feature_cnt):
        tf_idf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tf_idf[corpus], num_features=feature_cnt)
        return tf_idf, index

    # стоит объединить с общим токенизатором
    def tokenize_one_new(self,dictionary:dict) -> list:
        kw_vector = dictionary.doc2bow(tokenize(self.news_to_check))
        return kw_vector

    def compare_news(self,data,tf_idf, index,kw_vector):
        data['compare'] = index[tf_idf[kw_vector]]
        sort_similar_texts = data.sort_values(by='compare', ascending=False, ignore_index=True)[
            ['id', 'title', 'compare']]

        # print(sort_similar_texts)
        # print(sort_similar_texts.iloc[1]['title'])
        return sort_similar_texts.iloc[1]['title']







if __name__ == '__main__':
    test_new = 'Город в Гондурасе признал биткоин законным средством платежа'

    sim = TextSimilarity('test_file_db.csv',test_new)
    # сейчас мы каждый раз высчитываем все векторные представления по новой
    most_relevant_news = sim.get_similar_news()


    print('Новость: ' + test_new )
    print('Наиболее похожая на нее: ' + most_relevant_news)