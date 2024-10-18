import re
from typing import List, Union
import textwrap
import time
import os
import logging
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from AD_AUTO_GPT_functions import scrape_text, scrape_links, scrape_place_text, get_summary_period, text_all_lda, \
    get_city_info
from requests.packages import urllib3
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from langchain.prompts import StringPromptTemplate
from langchain.llms.base import BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import shutil

# Set up logging for debug purposes
logging.basicConfig(level=logging.ERROR)

# Load environment variables
load_dotenv()

# Model and Tokenizer Initialization with Hugging Face Transformers
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Using a smaller LLaMA2 model for testing
logging.info(f"Loading model: {model_name}")

try:
    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    logging.info("Model and tokenizer loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    exit(1)

#
# # Helper function to generate response using LLaMA
# def generate_llama_response(prompt: str) -> str:
#     try:
#         # Running on CPU for now, you can switch to GPU if it's available
#         logging.debug(f"Generating response for prompt: {prompt}")
#         inputs = tokenizer(prompt, return_tensors="pt")  # Running on CPU by default
#         logging.debug(f"Inputs prepared: {inputs}")
#
#         # Generate output from the model
#         outputs = model.generate(**inputs, max_new_tokens=500)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logging.debug(f"Model response: {response}")
#         return response
#     except Exception as e:
#         logging.error(f"Error during model generation: {e}")
#         return ""


def generate_llama_response(prompt: str) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate output with controlled randomness and a limit on the tokens
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # Limit token generation
            temperature=0.7,  # Lower temperature for faster, more focused response
            top_p=0.9,  # Top-p sampling
            top_k=50  # Top-k sampling
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Model response: {response}")
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


class ADGPT:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def google_search(self, query: str) -> str:
        urls = []
        pwd = os.getcwd()

        result = " "
        try:
            with DDGS() as ddgs:
                result = ddgs.text(query, max_results=20)

        except Exception as e:
            logging.error(f"An error occurred: {e}")

        if result is None:
            pwd = os.getcwd()
            news_links = os.listdir(pwd + '\\news_happend_lastyear\\')
            for news_link in news_links:
                shutil.copyfile(pwd + '\\news_happend_lastyear\\' + news_link, pwd + '\\workplace\\' + news_link)
            return "Internet error, but there are some news links stored on this device to help you know what is new to Alzheimer's disease research"
        with open(pwd + "/workplace/news_links.txt", "w", encoding='utf8') as file:
            for web in result:
                url = web['href']
                file.write(url + "\n")
                urls.append(url)
        return "The latest news has been saved on this device, you can use them to get what you want to know"

    def draw_news(self, query) -> str:
        pwd = os.getcwd() + '/workplace/'
        save_dir = pwd + 'news_summary/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        text_result = get_summary_period(pwd, save_dir)
        text_all_lda(text_result, save_dir)
        get_city_info(save_dir)
        return "Everything you need is obtained"

    def summary_news(self, query) -> str:
        pwd = os.getcwd()
        if not os.path.isdir(pwd + "/workplace"):
            os.mkdir(pwd + "/workplace")
        Dir = pwd + "/workplace/"

        news_web_list = ['CNN', 'Fox', 'Hill', 'NPR', 'USAToday']
        for web in news_web_list:
            urls = []

            count = 0
            with open(pwd + "/workplace/news_links.txt", "r", encoding='utf8') as file:
                urls = file.readlines()[:20]

            for url in urls:
                count += 1
                logging.info(f'Browsing {str(url)} and saving useful information in the workplace folder...')
                if not os.path.isdir(Dir + web):
                    os.mkdir(Dir + web)
                if not os.path.isdir(Dir + web + "/news_" + str(count)):
                    os.mkdir(Dir + web + "/news_" + str(count))
                    Dir1 = Dir + web + "/news_" + str(count)
                    text, datetime, news_title = scrape_text(url.replace('\n', ''), web)
                    with open(Dir1 + "/text.txt", "w", encoding='utf8') as f:
                        f.write(text)
                    cities = scrape_place_text(text)
                    with open(Dir1 + "/places.txt", "w", encoding='utf8') as f:
                        for city in cities:
                            f.writelines(city + '\n')
                    links = scrape_links(url)
                    with open(Dir1 + "/links.txt", "w", encoding='utf8') as f:
                        for link in links:
                            f.writelines(link + '\n')
                    if datetime != 0:
                        with open(Dir1 + "/dates.txt", "w", encoding='utf8') as f:
                            f.writelines(datetime + '\n')
                    with open(Dir1 + "/news_title.txt", "w", encoding='utf8') as f:
                        f.writelines(news_title + '\n')
                    with open(Dir1 + "/text.txt", "r", encoding='utf8') as f:
                        state_of_the_union = f.read()
                    texts = text_splitter.split_text(state_of_the_union)
                    docs = [Document(page_content=t) for t in texts[0:3]]
                    summary_file = " ".join([generate_llama_response(t) for t in texts[:3]])
                    with open(Dir1 + "/summary.txt", "w", encoding='utf8') as f:
                        f.write(summary_file)

        return "The news information you need is obtained, and the summary information is stored under the workplace folder"

    def introduce_info(self, query: str) -> str:
        """Introduce AD-GPT"""
        context = """
        With the powerful reasoning ability of LLM, we have built an automated task system similar to AutoGPT, \
            called AD-GPT, to collect and organize information related to AD on a daily basis. \
                AD-GPT has search, summary, storage, and drawing capabilities. With our own tools, \
                    it can automatically run relevant tasks and organize them without human intervention.
        """
        return generate_llama_response(context)


# Main function to run the LLaMA-based agent
if __name__ == "__main__":
    urllib3.disable_warnings()

    ad_gpt = ADGPT(llm=model)

    while True:
        try:
            print("Waiting for user input...")  # Debugging print
            user_input = input("Please enter your question/input: ")
            # logging.debug(f"User input received: {user_input}")

            if user_input.lower() == "exit":
                print("Exiting the program.")
                break

            try:
                response = generate_llama_response(user_input)
                # logging.debug(f"Generated response: {response}")
            except Exception as e:
                # logging.error(f"Error in LLaMA response generation: {e}")
                print(e)
                # continue

            try:
                output_response(response)
            except Exception as e:
                # logging.error(f"Error in output response: {e}")
                print(e)

        except KeyboardInterrupt:
            print("Program interrupted by user.")
            break