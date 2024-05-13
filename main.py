import glob
import os
from collections import Counter

import nltk
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from tqdm.auto import tqdm

from FetchArticles import createfile, delete_all_files

nltk.download('punkt')
nltk.download('stopwords')

TOP_N = 30  # You can set this to any number you want
REPORT_DIR = "generated_report"

# Ensure the report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_report(text_files):
  delete_all_files(REPORT_DIR)
  stop_words = set(stopwords.words('english'))
  all_tokens = []
  all_bigrams = []
  all_trigrams = []

  for file in tqdm(text_files, desc="Processing Files: "):
    with open(file, 'r') as f:
      text = f.read()
      tokens = [
          token.lower() for token in word_tokenize(text) if token.isalpha()
      ]
      filtered_tokens = [token for token in tokens if token not in stop_words]
      bigrams = list(ngrams(filtered_tokens, 2))
      trigrams = list(ngrams(filtered_tokens, 3))

      all_tokens.extend(filtered_tokens)
      all_bigrams.extend(bigrams)
      all_trigrams.extend(trigrams)

      report_filename = os.path.join(
          REPORT_DIR,
          os.path.basename(file).replace('.txt', '_report.txt'))

      with open(report_filename, 'w') as report_file:

        # Top N LSI keywords for individual article
        dictionary = Dictionary([filtered_tokens])
        corpus = [dictionary.doc2bow(text) for text in [filtered_tokens]]
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=1)
        top_topics = lsi.print_topics(num_words=TOP_N)
        report_file.write(f"\nTop {TOP_N} LSI keywords:\n")
        list_of_elements = [
            element for tuple in top_topics for element in tuple
        ]

        non_int_elements = [
            x for x in list_of_elements if not isinstance(x, int)
        ]
        x = ([i.split('+') for i in non_int_elements])
        answer = ([i.split('*') for i in x[0]])

        for i in answer:
          report_file.write(f"{i[1]}\n")

        # Top N frequent keywords
        freq_dist = FreqDist(filtered_tokens)
        report_file.write(f"\nTop {TOP_N} most frequent keywords:\n")

        for i in freq_dist.most_common(TOP_N):
          report_file.write(f"{i[0]}   --> {i[1]}\n")

        # Top N frequent bigrams
        bigram_counts = Counter(bigrams)
        report_file.write(f"\nTop {TOP_N} most frequent two word phrases:\n")
        for i in bigram_counts.most_common(TOP_N):
          report_file.write(f"{' '.join(i[0])}   --> {i[1]}\n")

        # Top N frequent trigrams
        trigram_counts = Counter(trigrams)
        report_file.write(f"\nTop {TOP_N} most frequent three word phrases:\n")
        for i in trigram_counts.most_common(TOP_N):
          report_file.write(f"{' '.join(i[0])}   --> {i[1]}\n")

  combined_report_filename = os.path.join(REPORT_DIR, 'combined_report.txt')

  with open(combined_report_filename, 'w') as combined_report_file:

    # Top N LSI keywords for all articles
    dictionary = Dictionary([all_tokens])
    corpus = [dictionary.doc2bow(text) for text in [all_tokens]]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=1)
    top_topics = lsi.print_topics(num_words=TOP_N)
    combined_report_file.write(f"\nTop {TOP_N} LSI keywords:\n")
    list_of_elements = [element for tuple in top_topics for element in tuple]

    non_int_elements = [x for x in list_of_elements if not isinstance(x, int)]
    x = ([i.split('+') for i in non_int_elements])
    answer = ([i.split('*') for i in x[0]])

    for i in answer:
      combined_report_file.write(f"{i[1]}\n")

    # Top N frequent keywords for all articles
    freq_dist_all = FreqDist(all_tokens)
    combined_report_file.write(f"\nTop {TOP_N} most frequent keywords:\n")

    for i in freq_dist_all.most_common(TOP_N):
      combined_report_file.write(f"{i[0]}   --> {i[1]}\n")

    # Top N frequent bigrams for all articles
    bigram_counts_all = Counter(all_bigrams)
    combined_report_file.write(
        f"\nTop {TOP_N} most frequent two word phrases:\n")
    for i in bigram_counts_all.most_common(TOP_N):
      combined_report_file.write(f"{' '.join(i[0])}   --> {i[1]}\n")

    # Top N frequent trigrams for all articles
    trigram_counts_all = Counter(all_trigrams)
    combined_report_file.write(
        f"\nTop {TOP_N} most frequent three word phrases:\n")
    for i in trigram_counts_all.most_common(TOP_N):
      combined_report_file.write(f"{' '.join(i[0])}   --> {i[1]}\n")


if __name__ == "__main__":
  urls = [
      'https://www.weblineindia.com/blog/types-of-mobile-app-development-services/',
      'https://www.weblineindia.com/blog/hire-offshore-python-developers/',
      'https://www.weblineindia.com/blog/outsource-dotnet-development-weblineindia/'
  ]
  text_files = createfile(urls)
  generate_report(text_files)
