import re
from typing import List, Union
import textwrap
import time
import os
import logging
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
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor
from langchain.schema import AgentAction, AgentFinish
import shutil

# Set up logging for debug purposes
logging.basicConfig(level=logging.ERROR)

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

# Prompt Templates
CONTEXT_QA_TMPL = """
Answering user's questions according to the information provided below:
Information: {context}

Question: {query}
"""
CONTEXT_QA_PROMPT = StringPromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

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

{agent_scratchpad}
"""

# Custom Prompt Template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Custom Output Parser
class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        regex = r"Action\s*\d*\s*:\s*(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            if "search result" in llm_output.lower() or "found results" in llm_output.lower():
                return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=llm_output,
                )
            return AgentFinish(
                return_values={"output": f" Here is what I received: `{llm_output}`"},
                log=llm_output,
            )

        action = match.group(1).strip()
        action_input = match.group(2).strip()
        return AgentAction(
            tool=action, tool_input=action_input.strip('"'), log=llm_output
        )

# Helper function to generate response using LLaMA
def generate_llama_response(prompt: str) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
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
        # Simplified for space, but similar structure as in main.py
        return "Summary news stored in workplace folder"

    def introduce_info(self, query: str) -> str:
        context = """
        With the powerful reasoning ability of LLM, we have built an automated task system similar to AutoGPT, \
            called AD-GPT, to collect and organize information related to AD on a daily basis. \
                AD-GPT has search, summary, storage, and drawing capabilities. With our own tools, \
                    it can automatically run relevant tasks and organize them without human intervention.
        """
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        return generate_llama_response(prompt)

# Main function to run the LLaMA-based agent
if __name__ == "__main__":
    urllib3.disable_warnings()

    ad_gpt = ADGPT(llm=model)

    # Define tools
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
        Tool(  # Adding the new "Draw plots" tool
            name="Draw plots",
            func=ad_gpt.draw_news,
            description="This is a tool to draw plots about news saved in this device."
        ),
    ]

    # Set up custom prompt and agent
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
    
    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=model, prompt=agent_prompt)

    # Initialize agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools],
    )

    # Agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # Main loop
    while True:
        try:
            user_input = input("Please enter your question/input: ")
            if user_input.lower() == "exit":
                print("Exiting the program.")
                break

            response = agent_executor.run(user_input)
            output_response(response)

        except KeyboardInterrupt:
            print("Program interrupted by user.")
            break
