import datetime
import logging
import os
import re
import textwrap
import time
from datetime import date, timedelta
from typing import Union, List, Tuple
from urllib.parse import urljoin
import os
import logging
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import gensim
import geopandas as gpd
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import requests
import spacy
import urllib3
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from gensim import corpora
from gensim.utils import simple_preprocess
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from geotext import GeoText
from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFacePipeline
from langchain.llms.base import BaseLLM
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from shapely.geometry import Point
from spacy.lang.en import English
from transformers import pipeline

# NLTK and SpaCy setup
spacy_model = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

# Setting up logging
logging.basicConfig(level=logging.ERROR)

stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know',
                   'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see',
                   'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even',
                   'right', 'line', 'even', 'also', 'may', 'take', 'come'])

proxies = None


def find_files(path, A):
    results = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == A + '.txt':
                results.append(os.path.join(root, name))
    return results


def get_city_info(save_path):
    pwd = os.getcwd()
    places_all = []
    geolocator = Nominatim(user_agent="Icarus", timeout=2)
    files = find_files(os.path.join(pwd, 'workplace'), 'places')

    for p in files:
        name = p.split('\\')[-3].split("/")[-1]

        with open(p, "r", encoding='utf8') as f:
            for places in f.readlines():
                location = geolocator.geocode(places.strip())
                if location:
                    places_all.append([
                        places.strip(),
                        name,
                        location.address,  # Changed to use 'address' instead of index
                        (location.latitude, location.longitude)
                    ])

    df = pd.DataFrame(places_all, columns=['City Name', 'News_Source', 'Country', 'Coordinates'])
    df.to_csv(os.path.join(save_path, 'geo_information.csv'), index=None)

    geometry = [Point(x[1], x[0]) for x in df['Coordinates']]
    crs = {'init': 'epsg:4326'}
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    countries_map = gpd.read_file(os.path.join(pwd, 'world_map.shp'))

    f, ax = plt.subplots(figsize=(16, 16))
    countries_map.plot(ax=ax, alpha=0.4, color='grey')

    # Define color mapping
    color_map = {
        'AA': '#C62828',  # Red
        'Mayo': '#283593',  # Dark Blue
        'bbc': '#FF9800',  # Orange
        'NIA': '#82B0D2',  # Light Blue
        'USAToday': '#FFC107',  # Amber
        'NPR': '#4CAF50',  # Green
        'Hill': '#FF5722',  # Deep Orange
        'Fox': '#9C27B0',  # Purple
        'CNN': '#2196F3',  # Blue
    }

    # Create a color column based on the mapping
    geo_df['color'] = geo_df['News_Source'].map(color_map).fillna('blue')  # Default color if not found

    # Plot the points with the corrected parameter
    geo_df['geometry'].plot(ax=ax, markersize=30, c=geo_df['color'], marker='^', alpha=0.5)

    font_dict = dict(fontsize=24, family='Times New Roman', style='italic')
    plt.title("Where Latest Alzheimer's Disease News Happen", fontdict=font_dict)
    plt.savefig(os.path.join(save_path, 'Places.jpg'), dpi=300)


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def prepare_text_for_lda(text):
    en_stop = set(nltk.corpus.stopwords.words('english'))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def get_time_difference(date_txt, news_type='bbc') -> int:
    time_now = datetime.datetime.now()
    date_desired = None

    if isinstance(date_txt, list):
        if date_txt:
            date_txt = date_txt[0]

    try:
        if news_type in ['bbc', 'NIA']:
            date_desired = datetime.datetime.strptime(date_txt, '%Y-%m-%d')
        elif news_type in ['AA', 'Mayo']:
            if len(date_txt.split("-")) <= 1:
                date_desired = datetime.datetime.strptime(date_txt, '%B %d, %Y')
            else:
                date_desired = datetime.datetime.strptime(date_txt, '%Y-%m-%d')

        if date_desired:
            day_difference = (time_now - date_desired).days
            return max(day_difference, 0)

    except ValueError as e:
        print(f"Error parsing date '{date_txt}': {e}")

    return 0


def find_files(path, A):
    results = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == A + '.txt':
                results.append(os.path.join(root, name))
    return results


def get_summary_period(path_dir, save_dir):
    result = []

    if os.path.exists(os.getcwd() + '\\workplace\\AA\\'):
        news_path = os.getcwd() + '\\workplace\\AA\\'
        newslist = os.listdir(news_path)
        count = 0
        for news in newslist:
            count = int(news.split('_')[1])
            if (not (os.path.exists(news_path + news + '/dates.txt'))):
                print(news)
                # time_now = datetime.datetime.now()
                date_desired1 = (date.today() - timedelta(days=int(365 * count / 140)))
                # print(date_desired1.strftime('%B %d, %Y'))
                with open(news_path + news + "/dates.txt", "w", encoding='utf8') as f:
                    f.writelines(str(date_desired1) + '\n')
    files = find_files(path_dir, 'dates')
    month_num = 12
    re_l = np.zeros(month_num)
    list_text_month = []
    X_label = []
    for i in range(0, month_num):
        list_text_month.append(str())
        X_label.append(str(i + 1))
    for date_f in files:
        with open(date_f, "r", encoding='utf8') as f:
            # print(date_f)
            date_txt = f.readline().replace('\n', '')
            # print(date_f)
            news_type = date_f.split("\\")[-3].split("/")[-1]
        summary_f = date_f.replace('dates', 'summary')
        with open(summary_f, "r", encoding='utf8') as f:
            summary_txt = f.readline().replace('\n', '')
        day_difference = get_time_difference(date_txt, news_type)
        time_index = int(day_difference / 30)
        if time_index >= month_num:
            time_index = month_num - 1
        re_l[time_index] += 1
        list_text_month[time_index] = list_text_month[time_index] + summary_txt
    X = np.arange(month_num)
    plt.figure(figsize=(12, 10))
    plt.bar(2.4 * X, re_l, color='#63b2ee', linewidth=1.5, width=1)
    X1 = ['2023-5', '2023-4', '2023-3', '2023-2', '2023-1', '2022-12', '2022-11', '2022-10', '2022-9', '2022-8',
          '2022-7', '2022-6']
    plt.xticks(ticks=2.4 * X, labels=X1, rotation=30, fontsize=20)
    plt.xlabel('Timeline (month)', fontsize=20)
    plt.ylabel('News Count', fontsize=20)
    plt.title('The Number of Relevant News that Happened in the Past Period',
              fontdict=dict(fontsize=24, family='Times New Roman', style='italic'))
    plt.savefig(save_dir + 'news_distribution_last_year.jpg', dpi=300)
    plt.close()
    result = list_text_month
    return result


def text_all_lda(text_result, save_path):
    topic_num_all = [4, 3, 3, 4, 3, 3, 3, 3, 3, 3, 4, 4]

    df_final = []
    K = []
    for i in range(12):
        K.append(str(i + 1) + '_Month_Keywords')
    text_all = str()
    for text_data, k, topic_num in zip(text_result, K, topic_num_all):
        if len(text_data) > 0:
            text_all = text_all + text_data
            data_words = prepare_text_for_lda(text_data)
            # print(data_words)

            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN']):  ##'VERB', 'ADJ', , 'ADV'
                """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
                texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
                texts = [bigram_mod[doc] for doc in texts]
                texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
                texts_out = []
                nlp = spacy.load("en_core_web_sm")
                # nlp = spacy.load('en', disable=['parser', 'ner'])
                for sent in texts:
                    doc = nlp(" ".join(sent))
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                # remove stopwords once more after lemmatization
                texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in
                             texts_out]
                return texts_out

            data_ready = process_words(data_words)  # processed Text Data!
            # Create Dictionary
            id2word = corpora.Dictionary(data_ready)

            # Create Corpus: Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in data_ready]

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=id2word,
                                                        num_topics=topic_num,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=10,
                                                        passes=10,
                                                        alpha='symmetric',
                                                        iterations=100,
                                                        per_word_topics=True)

            from collections import Counter
            topics = lda_model.show_topics(formatted=False)
            data_flat = [w for w_list in data_ready for w in w_list]
            counter = Counter(data_flat)

            out = []
            for i, topic in topics:
                for word, weight in topic:
                    out.append([word, i, weight, counter[word]])

            df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
            topic_words = {}
            for i in range(0, topic_num):
                topic_words.update(dict(topics[i][1]))
            # print(topic_words)
            np.save(save_path + 'topic_words_in_' + k + '.npy', topic_words)
            df['Time'] = k
            # df.to_csv('D:\\23spring\\AD-GPT\\workplace\\topic_words_in_'+k+'.csv', index =None)
            df_final.append(df)
            # pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
            pyLDAvis.save_html(vis, save_path + 'topic_words_in_' + k + '.html')
    df_result = pd.concat(df_final)
    df_result.to_csv(save_path + 'Topics_Trend_All.csv', index=None)
    data_words = prepare_text_for_lda(text_all)
    # print(data_words)
    num_topics_1 = 5
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_ready = process_words(data_words)  # processed Text Data!
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics_1,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)
    from collections import Counter
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
    topic_words = {}
    for i in range(0, num_topics_1):
        topic_words.update(dict(topics[i][1]))
    # print(topic_words)
    df.to_csv(save_path + 'topic_words_in_all_text.csv', index=None)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    pyLDAvis.save_html(vis, save_path + 'topic_words_in_all_text.html')


def scrape_text(url: str, news_url='bbc') -> str:
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text, datetime value, and news title
    """

    page = requests.get(url, verify=False)  # Ensure proxies are defined if needed
    soup = BeautifulSoup(page.content, features="html.parser")

    datetime_value = ""
    news_title = ""

    # Extract datetime value based on the news source
    if news_url in ['bbc', 'NIA']:
        time_element = soup.find('time')
        datetime_value = str(time_element.get('datetime')).split("T")[0] if time_element else "Unknown date"

    elif news_url == 'AA':
        time_element = soup.find("div", class_="metaDate")
        if time_element:
            contents = str(time_element.contents)
            datetime_value = contents.split("['\\r\\n  ")[1].split("\\n")[0] if len(
                contents.split("['\\r\\n  ")) >= 2 else 'March 1, 2023'
        else:
            datetime_value = 'Unknown date'

    elif news_url == 'ARUK':
        time_element = soup.find("meta", {"property": "article:published_time"}, content=True)
        datetime_value = str(time_element["content"]).split("T")[0] if time_element else "Unknown date"

    elif news_url == 'Mayo':
        time_element = soup.find("span", class_="moddate")
        datetime_value = str(time_element.contents[0]) if time_element else "Unknown date"

    elif news_url == 'AE':
        time_element = soup.find("div", class_="fl-module fl-module-rich-text fl-node-5e6b8729db02f news_date")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"

    elif news_url == 'CNN':
        time_element = soup.find("div", class_="timestamp")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"
        pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, str(datetime_value))
        datetime_value = str(matches[0]) if matches else "No date found"

    elif news_url == 'Fox':
        time_element = soup.find("div", class_="article-date")
        if time_element and time_element.contents:
            datetime_value = str(time_element.contents[0])
            pattern = r"\w+ \d+,\s\d+"
            matches = re.findall(pattern, datetime_value)
            datetime_value = str(matches[0]) if matches else "No date found"
        else:
            datetime_value = "No date found"

    elif news_url == 'Hill':
        time_element = soup.find("section",
                                 class_="submitted-by | header__meta | text-transform-upper text-300 color-light-gray weight-semibold font-base desktop-only")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"
        pattern = r"\d{1,2}/\d{1,2}/\d{2}"
        matches = re.findall(pattern, str(datetime_value))
        datetime_value = str(matches[0]) if matches else "No date found"

    elif news_url == 'NPR':
        time_element = soup.find("span", class_="date")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"

        news_title_tag = soup.find("div", class_="storytitle")

        if news_title_tag:
            news_title_h1 = news_title_tag.find("h1")
            news_title = news_title_h1.contents[0] if news_title_h1 and news_title_h1.contents else "No Title Found"
        else:
            news_title = "No Title Found"


    elif news_url == 'USAToday':
        time_element = str(soup)
        pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, time_element)
        datetime_value = str(matches[0]) if matches else "No date found"
        news_title_tag = soup.find("title")
        news_title = news_title_tag.contents[0] if news_title_tag and news_title_tag.contents else ""

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    # Get the main text content from the page
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text, datetime_value, news_title


def format_hyperlinks(hyperlinks: List[Tuple[str, str]]) -> List[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]


def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
    return [
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]


def scrape_links(url: str) -> Union[str, List[str]]:
    """Scrape links from a webpage.

    Args:
        url (str): The URL to scrape links from.

    Returns:
       Union[str, List[str]]: The scraped links or an error message.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        # Sending GET request with headers
        page = requests.get(url, verify=False, headers=headers, proxies=proxies, timeout=10)
        page.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        # Check if there are too many redirects
        if page.history:
            for resp in page.history:
                print(f'Redirected from {resp.url} to {page.url}')

        # Parse the page content
        soup = BeautifulSoup(page.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Extract hyperlinks (assuming extract_hyperlinks is defined)
        hyperlinks = extract_hyperlinks(soup, url)

        # Format hyperlinks (assuming format_hyperlinks is defined)
        return format_hyperlinks(hyperlinks)

    except requests.exceptions.TooManyRedirects:
        return "Error: Too many redirects."
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


def scrape_place_text(text):
    geolocator = Nominatim(user_agent="Icarus", timeout=2)
    places = GeoText(text)
    cities = list(set(list(places.cities)))
    cities_out = []
    for city in cities:
        try:
            location = geolocator.geocode(city)
            if location:
                cities_out.append(city)
        except GeocoderTimedOut as e:
            print(str(city) + ' is not a city')
    return cities


# Model and Tokenizer Initialization
from transformers import LlamaForCausalLM, LlamaTokenizer
import logging
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"


def load_llama_model(model_name: str):
    try:
        # Check if we're in a test environment where model loading should be skipped
        if os.getenv('SKIP_MODEL_LOADING', 'false') == 'true':
            logging.info("Skipping model loading due to environment setting.")
            return None, None

        # Load the tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_name)

        # Check if a GPU is available
        if torch.cuda.is_available():
            device = "cuda"
            logging.info("CUDA device available. Loading model on GPU.")
        else:
            device = "cpu"
            logging.info("CUDA not available. Loading model on CPU.")

        # Load the model with the appropriate device map
        model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        logging.info("Model and tokenizer loaded successfully.")

        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        return None, None


# Load model
tokenizer, model = load_llama_model(model_name)


# Helper function for generating response using LLaMA
def generate_llama_response(prompt: str) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9, top_k=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logging.error(f"Error during model generation: {e}")
        return ""


# Function to output response with a typewriter effect
def output_response(response: str) -> None:
    if not response:
        logging.error("No response to display.")
        exit(0)
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # Add a delay of 0.1 seconds between each character
            print(" ", end="", flush=True)  # Add a space between each word
        print()  # Move to the next line after each line is printed
    print("----------------------------------------------------------------")


# ADGPT Class (Updated with scraping and plotting functionalities)
class ADGPT:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def google_search(self, query: str) -> str:
        """
        Search the latest Alzheimer's disease news and save the URLs in a file.
        """
        pwd = os.getcwd()
        urls = []

        try:
            with DDGS() as ddgs:
                result = ddgs.text(query, max_results=20)
        except Exception as e:
            logging.error(f"Error during DuckDuckGo search: {e}")
            return "Search failed."

        # If search results are available, save them
        if result:
            os.makedirs(os.path.join(pwd, 'workplace'), exist_ok=True)
            with open(os.path.join(pwd, "workplace", "news_links.txt"), "w", encoding='utf8') as file:
                for res in result:
                    url = res['href']
                    file.write(url + "\n")
                    urls.append(url)
            return "The latest news has been saved on this device."
        else:
            return "No search results found."

    def summary_news(self, query: str) -> str:
        """
        Summarize the news by scraping content from the saved links.
        """
        pwd = os.getcwd()
        Dir = os.path.join(pwd, "workplace")
        os.makedirs(Dir, exist_ok=True)

        # List of news sources
        news_web_list = ['CNN', 'Fox', 'Hill', 'NPR', 'USAToday']

        try:
            with open(os.path.join(Dir, "news_links.txt"), "r", encoding='utf8') as file:
                urls = file.readlines()[:20]

            for url in urls:
                url = url.strip()
                for web in news_web_list:
                    # Create directories for each news source
                    web_dir = os.path.join(Dir, web)
                    os.makedirs(web_dir, exist_ok=True)

                    count = len(os.listdir(web_dir)) + 1
                    news_dir = os.path.join(web_dir, f"news_{count}")
                    os.makedirs(news_dir, exist_ok=True)

                    text, datetime_value, news_title = scrape_text(url, web)

                    with open(os.path.join(news_dir, "text.txt"), "w", encoding='utf8') as f:
                        f.write(text)
                    with open(os.path.join(news_dir, "places.txt"), "w", encoding='utf8') as f:
                        places = scrape_place_text(text)
                        for place in places:
                            f.write(f"{place}\n")
                    with open(os.path.join(news_dir, "links.txt"), "w", encoding='utf8') as f:
                        links = scrape_links(url)
                        for link in links:
                            f.write(f"{link}\n")
                    with open(os.path.join(news_dir, "dates.txt"), "w", encoding='utf8') as f:
                        f.write(f"{datetime_value}\n")
                    with open(os.path.join(news_dir, "news_title.txt"), "w", encoding='utf8') as f:
                        f.write(f"{news_title}\n")

                    # Summarize the news content
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
                    texts = text_splitter.split_text(text)
                    docs = [Document(page_content=t) for t in texts[:3]]  # Taking first 3 chunks for summarization
                    chain = load_summarize_chain(self.llm, chain_type="map_reduce")
                    summary = chain.run(docs)

                    with open(os.path.join(news_dir, "summary.txt"), "w", encoding='utf8') as f:
                        f.write(summary)

            return "News summaries have been saved in the workplace folder."

        except FileNotFoundError:
            return "No news links found to summarize."
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return "Error occurred during summarization."

    def draw_news(self, query: str) -> str:
        """
        Generate plots and visualizations for the news summaries.
        """
        pwd = os.getcwd() + '/workplace/'
        save_dir = pwd + 'news_summary/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        text_result = get_summary_period(pwd, save_dir)
        text_all_lda(text_result, save_dir)
        get_city_info(save_dir)
        return "Everything you need is obtained."

    def introduce_info(self, query: str) -> str:
        context = """
        With the powerful reasoning ability of LLM, we have built an automated task system similar to AutoGPT, \
            called AD-GPT, to collect and organize information related to AD on a daily basis. \
                AD-GPT has search, summary, storage, and drawing capabilities. With our own tools, \
                    it can automatically run relevant tasks and organize them without human intervention.
        """
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        return generate_llama_response(prompt)


# Prompt Templates
CONTEXT_QA_TMPL = """
Answering user's questions according to the information provided below:
Information: {context}

Question: {query}
"""

AGENT_TMPL = """Answer the following questions in the given format. You can use the following tools:

{tools}

Format of the answer:
---
Question: The question to be answered
Thought: What should I do to answer the question
Action: Choose one tool from ”{tool_names}“
Action Input: Choose the input arguments that action requires
Observation: Choose the results returned by tools
... (The action of thinking/observation can repeat N times)
Thought: Now I've got the final answer
Final Answer: The final answer to the question
---

Now starting to answer the user's question:

Question: {input}

Action: {agent_scratchpad}
"""


# Custom Prompt Template and Output Parser (as provided earlier)
class CustomPromptTemplate(StringPromptTemplate):
    @property
    def _prompt_type(self) -> str:
        pass

    def format(self, **kwargs) -> str:
        # Extract the necessary input variables from the arguments
        query = kwargs.get("input", "")
        intermediate_steps = kwargs.get("intermediate_steps", [])
        tools = kwargs.get("tools", [])
        tool_names = ", ".join([tool.name for tool in tools])  # list of tool names

        # Construct the agent's scratchpad (intermediate steps used to reason)
        scratchpad = ""
        if intermediate_steps:
            for step in intermediate_steps:
                scratchpad += f"Thought: {step['thought']}\n"
                scratchpad += f"Action: {step['action']}\n"
                scratchpad += f"Observation: {step['observation']}\n"

        # Choose the correct template based on context
        if "context" in kwargs:
            # Use CONTEXT_QA_TMPL if context is provided
            context = kwargs.get("context", "")
            formatted_prompt = CONTEXT_QA_TMPL.format(query=query, context=context)
        else:
            # Use AGENT_TMPL if context is not provided (agent-based task)
            formatted_prompt = AGENT_TMPL.format(
                input=query,
                agent_scratchpad=scratchpad,
                tools=tools,
                tool_names=tool_names
            )

        return formatted_prompt.strip()


class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse the LLM output and determine whether it is an action to be taken
        or the final answer.

        Args:
            llm_output (str): The output from the language model to be parsed.

        Returns:
            Union[AgentAction, AgentFinish]: Returns either an AgentAction
            (indicating a tool to use) or AgentFinish (indicating the final answer).
        """
        # Split the output into lines
        lines = llm_output.strip().split('\n')

        # Initialize variables to store parsed values
        thought = None
        action = None
        action_input = None
        final_answer = None

        for line in lines:
            # Check for the final answer
            if "Final Answer:" in line:
                final_answer = line.split("Final Answer:")[-1].strip()
                return AgentFinish(return_values={"output": final_answer}, log=llm_output)

            # Check for the thought process
            if "Thought:" in line:
                thought = line.split("Thought:")[-1].strip()

            # Check for the action to be taken
            if "Action:" in line:
                action = line.split("Action:")[-1].strip()

            # Check for the input required for the action
            if "Action Input:" in line:
                action_input = line.split("Action Input:")[-1].strip()

        # If action and action input are parsed, return an AgentAction
        if action and action_input:
            return AgentAction(
                tool=action,
                tool_input=action_input,
                log=llm_output
            )

        # If no action or final answer is found, return None or raise an error
        raise ValueError("Could not parse LLM output correctly.")


CONTEXT_QA_PROMPT = CustomPromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# Load model
tokenizer, model = load_llama_model(model_name)

# Define the device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check if model and tokenizer were loaded successfully
if model and tokenizer:
    # Create a pipeline using Hugging Face's pipeline API
    generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1  # Use GPU if available
    )

    # Wrap the pipeline using LangChain's HuggingFacePipeline class
    llm = HuggingFacePipeline(pipeline=generation_pipeline)

    # Main script logic
    if __name__ == "__main__":
        urllib3.disable_warnings()

        ad_gpt = ADGPT(llm=llm)  # Use the wrapped LLM

        # Define tools with the scraping and plotting logic
        tools = [
            Tool(
                name="Search and save the latest Alzheimer's disease news",
                func=ad_gpt.google_search,
                description="Search the latest news about Alzheimer's disease and save the URLs in a file."
            ),
            Tool(
                name="Summarize the news",
                func=ad_gpt.summary_news,
                description="Summarize the news and save it."
            ),
            Tool(
                name="Introduce AD-GPT",
                func=ad_gpt.introduce_info,
                description="Introduce the AD-GPT system."
            ),
            Tool(
                name="Draw plots",
                func=ad_gpt.draw_news,
                description="This is a tool to draw plots about news saved in this device."
            ),
        ]

        agent_prompt = CustomPromptTemplate(
            template=AGENT_TMPL,  # Your prompt template structure
            input_variables=["input", "intermediate_steps", "tools", "tool_names", "agent_scratchpad"],
            tools=tools
        )

        # Initialize the agent with the ChatPromptTemplate and tools
        agent = create_structured_chat_agent(
            llm=llm,  # Use the wrapped Llama model
            tools=tools,  # List of tools the agent can use
            prompt=agent_prompt  # Use ChatPromptTemplate
        )

        # Initialize the AgentExecutor
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, verbose=True)

        # Main loop to get user input and provide responses
        while True:
            try:
                user_input = input("Please enter your question/input: ")
                if user_input.lower() == "exit":
                    print("Exiting the program.")
                    break

                # Execute the agent's response
                response = agent_executor.run(user_input)
                print(response)

            except KeyboardInterrupt:
                print("Program interrupted by user.")
                break
else:
    logging.error("Model or tokenizer loading failed.")
