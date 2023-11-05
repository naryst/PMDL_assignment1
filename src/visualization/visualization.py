# Import necessary libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tqdm.notebook import tqdm

# Define the path to the raw data directory
RAW_DATA_DIR = "../../data/raw/"

# Read the dataset from a TSV file
dataset = pd.read_csv(RAW_DATA_DIR + 'filtered.tsv', sep='\t', index_col=0)

# Print statistics about the dataset
print(f"Mean similarity of the text pairs - {dataset['similarity'].mean()}")
print(f"Mean length difference of the text pairs - {dataset['lenght_diff'].mean()}")
print(f"Mean reference toxicity - {dataset['ref_tox'].mean()}")
print(f"Mean translation toxicity - {dataset['trn_tox'].mean()}")
print('=' * 100)

# Create new columns for toxicity difference and absolute toxicity difference
dataset['tox_diff'] = dataset.ref_tox - dataset.trn_tox
dataset['tox_diff_abs'] = np.abs(dataset.ref_tox - dataset.trn_tox)

# Create histograms for toxicity difference and absolute toxicity difference
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(dataset['tox_diff'])
axes[0].grid()
axes[0].set_xlabel('Toxicity Difference')
axes[0].set_title('Distribution of Toxicity Difference\n between reference and translation')
axes[1].hist(dataset['tox_diff_abs'])
axes[1].grid()
axes[1].set_xlabel('$\|$Toxicity Difference$\|$')
axes[1].set_title('Distribution of $\|$Toxicity Difference$\|$\n between reference and translation')
plt.savefig('Toxicity_dist.svg')

# Create new columns for the number of words in reference and translation
dataset['reference_words'] = dataset.apply(lambda x: len(x['reference'].split()), axis=1)
dataset['translation_words'] = dataset.apply(lambda x: len(x['translation'].split()), axis=1)

# Create histograms for the number of words in reference and translation
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(dataset['reference_words'])
axes[0].set_yscale('log')
axes[0].grid()
axes[0].set_xlabel('Number of Words')
axes[0].set_title('Reference Length Distribution')

axes[1].hist(dataset['translation_words'])
axes[1].set_yscale('log')
axes[1].grid()
axes[1].set_xlabel('Number of Words')
axes[1].set_title('Translation Length Distribution')
plt.savefig('Text_len_dist.svg')

# Create subsets of samples with detoxification and toxification
detoxification_samples = dataset[dataset['ref_tox'] >= dataset['trn_tox']]
toxification_samples = dataset[dataset['ref_tox'] < dataset['trn_tox']]

# Calculate the percentage of samples in the initial dataset that are text detoxification and toxification
print(f'{len(detoxification_samples) / len(dataset)*100:.3f}% of samples in the initial dataset are text detoxification')
print(f'{len(toxification_samples) / len(dataset)*100:.3f}% of samples in the initial dataset are text toxification')
print('=' * 100)

# Print statistics about toxicity in detoxification samples
print(f"Mean reference toxicity in detoxification - {detoxification_samples['ref_tox'].mean()}")
print(f"Mean translation toxicity in detoxification - {detoxification_samples['trn_tox'].mean()}")
print('=' * 100)

# Print statistics about toxicity in toxification samples
print(f"Mean reference toxicity in toxification - {toxification_samples['ref_tox'].mean()}")
print(f"Mean translation toxicity in toxification - {toxification_samples['trn_tox'].mean()}")
print('=' * 100)

# Create a WordCloud of toxic words in the training set
comment_words = ' '.join(detoxification_samples['reference'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800, 
            background_color='white', 
            stopwords=stopwords, 
            min_font_size=10).generate(comment_words)

# Plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.savefig('Toxic_wordCloud_trainset.svg')

# Define the path to the T5 results file
T5_res_path = '../../notebooks/test_set_results/T5_paraphrased_resV2.csv'

# Read the test data from the T5 results file
test_data = pd.read_csv(T5_res_path)

# Create WordClouds before and after text detoxification
comment_words_before = ' '.join(test_data['en_toxic_comment'])
comment_words_after = ' '.join(test_data['T5_paraphrased'])
stopwords = set(STOPWORDS)

wordcloud_before = WordCloud(width=800, height=800, 
            background_color='white', 
            stopwords=stopwords, 
            min_font_size=10).generate(comment_words_before)

wordcloud_after = WordCloud(width=800, height=800, 
            background_color='white', 
            stopwords=stopwords, 
            min_font_size=10).generate(comment_words_after)

# Create subplots for WordClouds before and after
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot the WordCloud images
axes[0].imshow(wordcloud_before)
axes[0].axis("off")
axes[0].set_title('WordCloud before')

axes[1].imshow(wordcloud_after)
axes[1].axis("off")
axes[1].set_title('WordCloud after')

plt.savefig('WordClouds_compare_testset.svg')
