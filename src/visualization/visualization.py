import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tqdm.notebook import tqdm

RAW_DATA_DIR = "../../data/raw/"
dataset = pd.read_csv(RAW_DATA_DIR + 'filtered.tsv', sep='\t', index_col=0)

print(f"Mean simmilarity of the text paris - {dataset['similarity'].mean()}")
print(f"Mean length difference of the text paris - {dataset['lenght_diff'].mean()}")
print(f"Mean reference toxisity - {dataset['ref_tox'].mean()}")
print(f"Mean translation toxisity - {dataset['trn_tox'].mean()}")
print('='*100)


dataset['tox_diff'] = dataset.ref_tox - dataset.trn_tox
dataset['tox_diff_abs'] = np.abs(dataset.ref_tox - dataset.trn_tox)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].hist(dataset['tox_diff'])
axes[0].grid()
axes[0].set_xlabel('Toxisity diff')
axes[0].set_title('Distribution of Toxisity diff \n between reference and translation')
axes[1].hist(dataset['tox_diff_abs'])
axes[1].grid()
axes[1].set_xlabel('$\|$Toxisity diff$\|$')
axes[1].set_title('Distribution of $\|$Toxisity diff$\|$ \n between reference and translation')
plt.savefig('Toxisity_dist.svg')

dataset['reference_words'] = dataset.apply(lambda x: len(x['reference'].split()), axis=1)
dataset['translation_words'] = dataset.apply(lambda x: len(x['translation'].split()), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(dataset['reference_words'])
axes[0].set_yscale('log')
axes[0].grid()
axes[0].set_xlabel('number of words')
axes[0].set_title('Reference length distribution')

axes[1].hist(dataset['translation_words'])
axes[1].set_yscale('log')
axes[1].grid()
axes[1].set_xlabel('number of words')
axes[1].set_title('Translation length distribution')
plt.savefig('Text_len_dist.svg')

detoxification_samples = dataset[dataset['ref_tox'] >= dataset['trn_tox']]
toxification_samples = dataset[dataset['ref_tox'] < dataset['trn_tox']]

print(f'{len(detoxification_samples) / len(dataset)*100:.3f}% of samples in the initial dataset is text detoxification')
print(f'{len(toxification_samples) / len(dataset)*100:.3f}% of samples in the initial dataset is text toxification')
print('='*100)

print(f"Mean reference toxisity in detox - {detoxification_samples['ref_tox'].mean()}")
print(f"Mean translation toxisity in detox - {detoxification_samples['trn_tox'].mean()}")
print('='*100)


print(f"Mean reference toxisity in toxification - {toxification_samples['ref_tox'].mean()}")
print(f"Mean translation toxisity in toxification - {toxification_samples['trn_tox'].mean()}")
print('='*100)



comment_words = ' '.join(detoxification_samples['reference'])

stopwords = set(STOPWORDS)

wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words) 

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.savefig('Toxic_wordCloud_trainset.svg')


T5_res_path = '../../notebooks/test_set_results/T5_paraphrased_resV2.csv'
test_data = pd.read_csv(T5_res_path)

comment_words_before = ' '.join(test_data['en_toxic_comment'])
comment_words_after = ' '.join(test_data['T5_paraphrased'])

stopwords = set(STOPWORDS)

wordcloud_before = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words_before)

wordcloud_after = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(comment_words_after)

fig, axes = plt.subplots(1, 2, figsize=(16,8))
# plot the WordCloud image
axes[0].imshow(wordcloud_before)
axes[0].axis("off")
axes[0].set_title('WordCloud before')

axes[1].imshow(wordcloud_after)
axes[1].axis("off")
axes[1].set_title('WordCloud after')

plt.savefig('WordClouds_compare_testset.svg')
