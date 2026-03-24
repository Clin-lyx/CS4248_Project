from pathlib import Path
import json

class DatasetLoader:
    def __init__(self, dataset_path="dataset/Sarcasm_Headlines_Dataset_v2.json"):
        self.dataset_path = Path(__file__).parent.parent / dataset_path
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def get_data(self):
        return self.data

    def get_data_size(self):
        return len(self.data)

    def get_data_sample(self, index):
        return self.data[index]