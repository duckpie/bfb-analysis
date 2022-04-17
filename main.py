# from src.config import Config
from src.analysis.news_similarity import TextSimilarity
import pandas as pd



if __name__ == '__main__':
    # конфиг читаем с консоли
    # cfg = Config('local')



    test_new = 'Цены на нефть завершили предпраздничные торги активным ростом'

    file_name = 'src/analysis/test_file_db.csv'
    data = pd.read_csv(file_name)

    for new in data.index:
        test_new = data.iloc[new]['title']

        sim = TextSimilarity(data, test_new)
        # сейчас мы каждый раз высчитываем все векторные представления по новой
        most_relevant_news = sim.get_similar_news()

        print('Новость: ' + test_new)
        print('Наиболее похожая на нее: ' + most_relevant_news)
        print('=====')



