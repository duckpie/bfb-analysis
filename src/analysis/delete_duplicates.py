import pandas as pd

class DeleteDublicates():
    def __init__(self,file_name):
        self.file_name = file_name

    def load_data(self):
        return pd.read_csv(self.file_name)

    def save_unique(self):
        data = self.load_data()
        no_duplicates = data.drop_duplicates()
        no_duplicates.to_csv(self.file_name, index=False)
        print('Дубликаты успешно удалены')
        print(f"Было: {data.shape[0]}")
        print(f"Стало: {no_duplicates.shape[0]}")



if __name__ == '__main__':
    no_dubl = DeleteDublicates('test_file_db.csv')
    no_dubl.save_unique()