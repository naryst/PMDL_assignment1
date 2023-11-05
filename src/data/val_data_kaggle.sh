cd ../../data/external/
gdown --fuzzy https://drive.google.com/file/d/1rUI3zkebeLjuSknixEF91cME5dNJeKe_/view?usp=drive_link
cd ../../src/data/
python kaggle_preprocessing.py "../../data/external/kaggle_toxic_data.csv" "../../data/external/"