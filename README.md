# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

### Nikita Sergeev BS20-AI n.sergeev@innopolis.university

## Project structure
```
text-detoxification
├── README.md # The top-level README
│
├── data 
│   ├── external # Data from third party sources|
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks    #  Jupyter notebooks. Naming convention is a number (for ordering),
│                   and a short delimited description, e.g.
│                   "1.0-initial-data-exporation.ipynb"            
│ 
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │   
    └── visualization   # Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```


In the top `README.md` file put your name, email and group number. Additionaly, put basic commands how to use your repository. How to transform data, train model and make a predictions.


#### Data preparation
To download and transform the training data into the appropriate format you should use the following bash script:
```bash
sh download_data.bash
```
You should execute this script from the `src/data` folder make all the paths correct.
After this, all the preprocessed data needed for trining models will be stored in `data/interim/hf_dataset`


#### Model training
Execute the following python script
```bash
python T5_train.py
```
This will train T5 model on the filtered ParaNMT dataset and store the trained model weights in the `models/T5_paraphraser/` dir.

#### Model evaluation
There is 3 different options for the model evaluation.
* Pretrained BART to detoxify the given text - `s-nlp/bart-base-detox`
* Manually Fine-Tuned T5 for toxic texts paraphraser
* Mask the toxic words in the given text with the `s-nlp/roberta_toxicity_classifier`

Results of the evaluation in the test set (`s-nlp/paradetox`) are available in the `TODO`


#### Model Inference
Where are 3 different options for the model inference
* Manually finetuned T5 - ```python T5_inference.py```
* Pretrained BART - ```python BART_inference.py```
* Toxic words masking - ```python toxic_words_masking.py```
