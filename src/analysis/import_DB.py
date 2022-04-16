# -*- coding: utf-8 -*-
import json
import pandas as pd

import redis
import psycopg2


class DataBase():

    def __init__(self, redis=None,db_cursor=None):
        self.redis = redis
        self.cursor = db_cursor

    def create_db(self):

        self.cursor.execute('''CREATE TABLE NEWS  
             (id CHAR(50) PRIMARY KEY NOT NULL,
             title TEXT NOT NULL,
             country CHAR(50),
             language CHAR(50),
             rights CHAR(50),
             clean_url TEXT);''')

        print("Table created successfully")



        print('Creating DB...')

    def get_data_from_redis(self):
        keys = self.redis.keys()
        values = self.redis.mget(keys)

        kv = zip(keys, values)

        df = pd.DataFrame(columns=['id','title','country','language','rights','clean_url'])

        for keys, values in kv:
            key = keys.decode("utf-8")
            value = json.loads(values.decode("utf-8"))

            value['id'] = key

            df = df.append(value,ignore_index=True)

        return df


    def save_data_to_file(self,data):
        print(data)

        data.to_csv('test_file_db.csv',index=False)
        print('Successfully saved')

    def save_data_to_db(self):
        with open('test_file_db.csv', 'r', encoding="utf-8") as f:
            next(f)
            self.cursor.copy_from(f, 'NEWS', sep=',')

        print('вроде как сохранилось')

        # s.execute('SELECT * FROM airport LIMIT 10')
        # records = cursor.fetchall()

    def check_db(self):
        self.cursor.execute('SELECT * FROM NEWS')
        res = self.cursor.fetchall()
        print(res)

    def work_with_redis(self):
        data = self.get_data_from_redis()
        self.save_data_to_file(data)

    def main(self):

        self.work_with_redis()


        # сохраняем данные в бд
        self.create_db()
        self.save_data_to_db()
        self.check_db()
        print('Complete')


if __name__ == '__main__':
    # тут не импортирован конфиг
    # cfg = Config('local')

    # r = redis.Redis(
    #     host="127.0.0.1",
    #     port=6379,
    #     db=2
    # )


    conn = psycopg2.connect(dbname='test_news_db', user='db_user',
                            password='pass', host="127.0.0.1")
    cursor = conn.cursor()



    # db = DataBase(, r)
    db = DataBase(cursor)
    db.main()

    conn.commit()
    cursor.close()

    conn.close()
