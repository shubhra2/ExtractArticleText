import string
import re
from pathlib import Path

import nltk
# from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from FetchArticles import createfile

nltk.download('punkt')
nltk.download('stopwords')


def is_valid_keyword(keyword):
  common_words = {
      'can', 'an', 'at', 'but', 'the', 'and', 'a', 'is', 'in', 'it', 'of',
      'to', 'for', 'on', 'with', 'as', 'by', 'that', 'be', 'am', 'are'
  }
  # Split the phrase into individual words
  words = keyword.split()
  # Check if each word is alphabetic and not a common word
  valid_words = [
      word for word in words
      if word.isalpha() and word.lower() not in common_words
  ]
  # The phrase is valid if it has the same number of valid words as the original
  return len(valid_words) == len(words)


def clean_text(text):
  text = text.lower()
  text = text.translate(str.maketrans("", "", string.punctuation))
  words = word_tokenize(text)
  words = [word for word in words if word not in stopwords.words('english')]
  return words


def get_lsi_keywords(text, num=10):
  documents = re.split(r'(?<=[.!?])\s+', text)
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(documents)
  cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

  average_similarities = cosine_similarities.mean(axis=0)
  indices = average_similarities.argsort()[-num:]
  feature_names = vectorizer.get_feature_names_out()

  keywords = [
      feature_names[index] for index in indices[::-1]
      if is_valid_keyword(feature_names[index])
  ]
  return keywords


def get_high_frequency_words(text, num=10):
  words = clean_text(text)
  words = [word for word in words if is_valid_keyword(word)]
  counter = Counter(words)
  most_common = counter.most_common(num)
  return most_common


def get_high_frequency_phrases(text, num=10):
  words = clean_text(text)
  # Construct candidate phrases from adjacent cleaned words
  phrases = [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
  # Filter out any non-valid phrases
  valid_phrases = [phrase for phrase in phrases if is_valid_keyword(phrase)]
  counter = Counter(valid_phrases)
  most_common = counter.most_common(num)
  return most_common


def get_high_frequency_three_phrases(text, num=10):
  words = clean_text(text)
  # Construct candidate phrases from adjacent cleaned words
  phrases = [" ".join(words[i:i + 3]) for i in range(len(words) - 1)]
  # Filter out any non-valid phrases
  valid_phrases = [phrase for phrase in phrases if is_valid_keyword(phrase)]
  counter = Counter(valid_phrases)
  most_common = counter.most_common(num)
  return most_common


def main(urls):
  # num_files = int(input("How many text files do you want to analyze? "))

  # Initialize empty lists to store aggregated results
  all_lsi_keywords = []
  all_high_freq_words = Counter()
  all_high_freq_phrases = Counter()
  all_high_freq_three_phrases = Counter()

  for filename in createfile(urls):
    # filename = input("Enter the name of the text file: ")
    with open(filename, 'r', encoding='utf-8') as file:
      text = file.read()

    lsi_keywords = get_lsi_keywords(text)
    high_freq_words = get_high_frequency_words(text)
    high_freq_phrases = get_high_frequency_phrases(text)
    high_freq_three_phrases = get_high_frequency_three_phrases(text)

    # Save individual analysis for the file
    only_file = Path(filename)
    with open(f"generated_report/analysis_{only_file.name}",
              "w",
              encoding='utf-8') as analysis_file:
      analysis_file.write(f"Analysis for file: {filename}\n")
      analysis_file.write("LSI Keywords:\n")
      for keyword in lsi_keywords:
        analysis_file.write(f"- {keyword}\n")

      analysis_file.write("\nHigh Frequency Keywords:\n")
      for keyword, count in high_freq_words:
        analysis_file.write(f"- {keyword}\n")

      analysis_file.write("\nHigh Frequency Two-Word Phrases:\n")
      for phrase, count in high_freq_phrases:
        analysis_file.write(f"- {phrase}\n")

      analysis_file.write("\nHigh Frequency Three-Word Phrases:\n")
      for phrase, count in high_freq_three_phrases:
        analysis_file.write(f"- {phrase}\n")

      # Update aggregated results
      all_lsi_keywords.extend(lsi_keywords)
      all_high_freq_words.update(high_freq_words)
      all_high_freq_phrases.update(high_freq_phrases)
      all_high_freq_three_phrases.update(high_freq_three_phrases)

    # Save combined analysis to "combined_analysis.txt"
    with open("generated_report/combined_analysis.txt", "w",
              encoding='utf-8') as analysis_file:
      analysis_file.write("Combined Analysis for All Text Files\n")

      analysis_file.write("\nLSI Keywords (Across All Files):\n")
      # Remove duplicates from aggregated LSI keywords before saving
      unique_lsi_keywords = list(set(all_lsi_keywords))
      for keyword in unique_lsi_keywords:
        analysis_file.write(f"- {keyword}\n")

      analysis_file.write("\nHigh Frequency Words (Across All Files):\n")
      for keyword, count in all_high_freq_words.most_common():
        analysis_file.write(f"- {keyword}\n")

      analysis_file.write(
          "\nHigh Frequency Two-Word Phrases (Across All Files):\n")
      for phrase, count in all_high_freq_phrases.most_common():
        analysis_file.write(f"- {phrase}\n")

      analysis_file.write(
          "\nHigh Frequency Three-Word Phrases (Across All Files):\n")
      for phrase, count in all_high_freq_three_phrases.most_common():
        analysis_file.write(f"- {phrase}\n")

      print("\nAnalysis results saved to individual and combined files.")

    # Empty the text file
    # with open(filename, 'w', encoding='utf-8') as empty_file:
    #   empty_file.write('')
    #   print(f"Text file {filename} emptied.")

  # print("\nAll text files emptied and analysis results saved.")


if __name__ == "__main__":
  urls = [
      'https://www.weblineindia.com/blog/types-of-mobile-app-development-services/',
      'https://www.weblineindia.com/blog/types-of-mobile-app-development-services/',
      'https://www.weblineindia.com/blog/outsource-dotnet-development-weblineindia/'
  ]
  main(urls)
