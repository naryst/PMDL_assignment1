import datasets
import os
import sys

def main():
    data_path = sys.argv[1]
    save_path = sys.argv[2]

    data = datasets.load_dataset("csv", data_files=data_path)
    data = data.filter(lambda x: x['toxic'] == 1)
    data = data.remove_columns(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    data = data['train']
    data.save_to_disk(save_path + "kaggle_val_dataset")


if __name__ == '__main__':
    main()