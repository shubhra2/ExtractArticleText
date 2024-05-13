import os
from newspaper import Article
from html2text import html2text
import glob
from tqdm.auto import tqdm

def delete_all_files(directory):
  # Ensure that the path is valid
  if not os.path.isdir(directory):
      print("The specified directory does not exist.")
      return

  for filename in os.listdir(directory):
      file_path = os.path.join(directory, filename)
      try:
          # Remove the file
          os.remove(file_path)
          # print(f"Deleted {filename}")
      except Exception as e:
          print(f"Error deleting {filename}: {e}")

def createfile(url):
  # flush the output directory
  delete_all_files('article_text')
  
  count = 0
  for count, i in enumerate(tqdm(url, desc="Downloading articles: ")):

    count += 1
    article = Article(i)
    article.download()
    article.parse()
    
    with open(f'article_text/output{count}_article.txt', 'w') as outfile:
      outfile.write(article.text)

  print("Total Files: ", count)
  text_files = glob.glob("article_text/*_article.txt")
  print(text_files)
  return text_files


if __name__ == "__main__":
  url = [
      'https://www.weblineindia.com/blog/types-of-mobile-app-development-services/',
      'https://www.weblineindia.com/blog/types-of-mobile-app-development-services/',
      'https://www.weblineindia.com/blog/outsource-dotnet-development-weblineindia/',
      'https://www.weblineindia.com/blog/prompt-engineering-in-software-development/'
  ]
  createfile(url)
