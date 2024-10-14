import re
from typing import List, Union
import textwrap
import time
import os 
from duckduckgo_search import DDGS
from AD_AUTO_GPT_functions import scrape_text, scrape_links, scrape_place_text, get_summary_period,text_all_lda,get_city_info
from requests.packages import urllib3
from langchain.agents import ( Tool, AgentExecutor, LLMSingleActionAgent, initialize_agent, AgentOutputParser)
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.llms.base import BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import AgentType
import shutil
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=0)

CONTEXT_QA_TMPL = """
Answer user's questions according to the information provided below
Information：{context}

Question：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()


def output_response(response: str) -> None:
    if not response:
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
       
       
        urls= []
        pwd = os.getcwd()
       
        re=" " 
        try:
            with DDGS() as ddgs:
                re = ddgs.text(query, max_results=20)
                
        except Exception as e:
            print(f"An error occurred: {e}")
            
        if(re==None):
            pwd = os.getcwd()
            news_links = os.listdir(pwd+'\\news_happend_lastyear\\')
            for news_link in news_links:
                shutil.copyfile(pwd+'\\news_happend_lastyear\\'+news_link, pwd+'\\workplace\\'+news_link)
            return "Internet error, but there are some news links stored on this device to help you know what is new to alzheimer's disease research"
        with open(pwd+"/workplace/news_links.txt", "w", encoding='utf8') as file:
            
            for web in re:
                url = web['href']
             
                file.write(url + "\n")
                urls.append(url) 
        return "The latest news has been saved on this device, you can use them to get what you want to know"
    
    def draw_news(self,query) -> str:
        pwd = os.getcwd() +'/workplace/'
        save_dir =  pwd + 'news_summary/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        text_result = get_summary_period(pwd,save_dir)
        text_all_lda(text_result,save_dir)
        get_city_info(save_dir)
        return  "every thing you need is obtained"     
           
    def summary_news(self,query) -> str:
        pwd = os.getcwd()
        if not os.path.isdir(pwd+"/workplace"):
            os.mkdir(pwd+"/workplace")
        Dir = pwd+"/workplace/"
        
        news_web_list = ['CNN','Fox','Hill','NPR','USAToday']
        # news_web_list = ['bbc']
        for web in news_web_list:
            urls = []
             
                 
            count = 0
            with open(pwd+"/workplace/news_links.txt", "r", encoding='utf8') as file:
                urls=file.readlines()[:20]
            
            for url in urls:
                count = count +1 
                print('\nBrowsing '+str(url)+'and save useful infomation in workplace folder...')
                if not os.path.isdir(Dir+web):
                    os.mkdir(Dir+web)
                if not os.path.isdir(Dir+web+"/news_"+str(count)):
                    os.mkdir(Dir+web+"/news_"+str(count))
                    Dir1 = Dir+web+"/news_"+str(count)
                    text,datetime,news_title = scrape_text(url.replace('\n',''),web)
                    with open(Dir1 + "/text.txt", "w", encoding='utf8') as f:
                        f.write(text)
                    cities = scrape_place_text(text)
                    with open(Dir1 + "/places.txt", "w", encoding='utf8') as f:
                        for city in cities:
                            f.writelines(city+'\n')     
                    links = scrape_links(url)
                    with open(Dir1 + "/links.txt", "w", encoding='utf8') as f:
                        for link in links:
                            f.writelines(link+'\n') 
                    if(datetime != 0): 
                        with open(Dir1 + "/dates.txt", "w", encoding='utf8') as f:
                            f.writelines(datetime+'\n')    
                    with open(Dir1 + "/news_title.txt", "w", encoding='utf8') as f:
                            f.writelines(news_title+'\n')                     
                    with open(Dir1 + "/text.txt", "r", encoding='utf8') as f:
                        state_of_the_union = f.read()
                    texts = text_splitter.split_text(state_of_the_union)
                    docs = [Document(page_content=t) for t in texts[0:3]]
                    chain = load_summarize_chain(llm, chain_type="map_reduce") 
                    summary_file = chain.run(docs)
                    with open(Dir1 + "/summary.txt", "w", encoding='utf8') as f:
                        f.write(summary_file)

        #Uncomment this line if you need the visualization
        """              
        pwd = os.getcwd() +'/workplace/'
        save_dir =  pwd + 'news_summary/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if os.path.exists(save_dir+"Topics_Trend_All.csv"):
            print("Visualizing the news topics...")
            text_result = get_summary_period(pwd,save_dir)  
            text_all_lda(text_result,save_dir)
        if os.path.exists(save_dir+"geo_information.csv"):
            print("Visualizing the news places...")
            get_city_info(save_dir)


        """
        return "The news information you need is obtained, the summary information is stored under the workplace folder"
    
    def introduce_info(self, query: str) -> str:
        """introduce AD-GPT"""
        context = """
        With the powerful reasoning ability of LLM, we have built an automated task system similar to AutoGPT, \
            called AD-GPT, to collect and organize information related to AD on a daily basis. \
                AD-GPT has search, summary, storage, and drawing capabilities. With our own tools, \
                    it can automatically run relevant tasks and organize them without human intervention.
        """
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        
        return self.llm(prompt)

AGENT_TMPL = """Answer the following questions in the given format, You can use the following tools：

{tools}

When answering, please follow the format enclosed in ---

---
Question: The question need to be answered
Thought: What should I do to answer the above question
Action: choose one tool from ”{tool_names}“ 
Action Input: choose the input_args that action requires
Observation: Choose the results returned by tools
...（The action of thinking/observation can repeat N times）
Thought: Now, I've got the final answer
Final Answer: The final answer of the initial question
---

Now start to answer user's questions, remember to follow the specified format step by step before providing the final answer.

Question: {input}


{agent_scratchpad}
"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str  # Standard template
    tools: List[Tool]  # Usable tools

    def format(self, **kwargs) -> str:
        """
        Fill in all the necessary values according to the defined template.
        
        Returns:
            str: filled template。
        """
        intermediate_steps = kwargs.pop("intermediate_steps")  # Extract the intermediate steps and execute them.
        
        thoughts = ""
        for action, observation in intermediate_steps:
           
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts  # Record the thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )  # Enumerate all available tool names and tool descriptions
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )  # Enumerate all tools' names
        cur_prompt = self.template.format(**kwargs)
        print(cur_prompt)
        return cur_prompt
 

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Interpret the output of LLM and locate the necessary actions based on the output text.

        Args:
            llm_output (str): Output from the LLM, which could include actions or final answers.

        Raises:
            ValueError: Raised when output doesn't match the expected format.

        Returns:
            Union[AgentAction, AgentFinish]: Either an action to take or a final output.
        """
        # Check if it contains "Final Answer" to finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Regex to find actions in the format: Action X: [Action] / Action Input: [Input]
        regex = r"Action\s*\d*\s*:\s*(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If no match, handle search-specific output or other unexpected output
        if not match:
            if "search result" in llm_output.lower() or "found results" in llm_output.lower():
                # Assume this is a search result or a similar response
                return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=llm_output,
                )

            # Handle specific output indicating successful completion
            if "now, i've got the final answer" in llm_output.lower():
                return AgentFinish(
                    return_values={"output": "Task completed successfully."},
                    log=llm_output,
                )

            # Return the original text with an uncertainty message
            return AgentFinish(
                return_values={"output": f" Here is what I received: `{llm_output}`"},
                log=llm_output,
            )

        action = match.group(1).strip()
        action_input = match.group(2).strip()

        # Handling specific actions like "Introduce AD-GPT"
        if action.lower() == "introduce ad-gpt":
            return AgentAction(
                tool=action, tool_input="AD-GPT", log=llm_output
            )

        return AgentAction(
            tool=action, tool_input=action_input.strip('"'), log=llm_output
        )



if __name__ == "__main__":
    ## set api token in terminal
    
    os.environ["OPENAI_API_KEY"] = "Add your key"    
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    urllib3.disable_warnings()
    ad_gpt = ADGPT(llm)
     
    tools = [
        Tool(
            name="Search and save the latest Alzheimer's disease news", 
            func=ad_gpt.google_search,
            description="This is a tool that use the Google to search for the latest news about Alzhemier's disease and save the URLs in a file",
        ),
        Tool(
            name="Summarise the news",
            func=ad_gpt.summary_news,
            description="This is a tool to know when and where the   happens,which will extract and save the time, place and hyperlinks in the news.",
        ),
        Tool(
            name="Introduce AD-GPT",
            func=ad_gpt.introduce_info,
            description="This is a tool to introduce AD-GPT",
        ),
         Tool(
            name="Draw plots",
            func=ad_gpt.draw_news,
            description="This is a tool to draw plots about news saved in this device",
        ),
    ]
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
   
    output_parser = CustomOutputParser()

    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
   
    tool_names = [tool.name for tool in tools]
    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
  
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
     
    while True:
        try:
            user_input =input("please enter the input  :")
    
            response = agent_executor.run(user_input)
            
 
            output_response(response)
        except KeyboardInterrupt:
            break