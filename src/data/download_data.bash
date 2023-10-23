cd ../../data/raw
echo Installing the filtered ParaNMT-detox corpus into $(pwd)
wget https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip >> /dev/null 2>&1
unzip -u filtered_paranmt.zip
rm filtered_paranmt.zip 
echo Done, raw dataset installed as $(pwd)/filtered_paranmt.tsv
echo Start processing raw data and converting it to HuggingFace Dataset
cd ../../src/data/
python data_loading.py ../../data/raw/ ../../data/interim/