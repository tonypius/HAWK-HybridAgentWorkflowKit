# app.py
from typing import List, Union, Optional
import langchain
langchain.verbose=False
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp, HuggingFaceTextGenInference
from langchain.embeddings import LlamaCppEmbeddings, SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant, Chroma
# from PyPDF2 import PdfReader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
# import streamlit as st
import tempfile
# import PyPDF2

import os
os.environ["OPENAI_API_KEY"] = "openapi-key-here"
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.schema import Document
from langchain.llms import OpenAI

from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")

from langchain.chat_models import AzureChatOpenAI

os.environ["AZURE_OPENAI_KEY"] = "openapi-key-here"
os.environ["AZURE_OPENAI_ENDPOINT"] = "endpoint-here"
verbose = False

BASE_URL = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT_NAME = 'deployment-name-here' #This will correspond to the custom name you chose for your deployment when you deployed a model.
OPENAPI_VERSION = "2023-05-15"
llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=OPENAPI_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)

# openai.temperature = 0.1


# Set up the base template
TEMPLATE = """Answer the following questions as an Expert and answer it always based on the given tools. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}] 
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as an Expert who can answer from the given tools when giving your final answer.

Question: {input}
{agent_scratchpad}"""


def qa_tools_for_else(embeddings):

    retriever_bank = Chroma(persist_directory='./bank',embedding_function=embeddings)
    retriever_bank.persist()
    bank_data = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever_bank.as_retriever()
    )

    retriever_esg = Chroma(persist_directory='./esg',embedding_function=embeddings)
    retriever_esg.persist()
    esg_data = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever_esg.as_retriever()
    )

    tools = [
        Tool(
            name="Bank",
            func=bank_data.run,
            description="useful for when you need to answer questions about the banking related questions. Input should be a fully formed question.",
        ),
        Tool(
            name="ESG",
            func=esg_data.run,
            description="useful for when you need to answer questions about ESG. Input should be a fully formed question.",
        ),
    ]
    return tools

def get_toolsdesc(query,retriever,ALL_TOOLS):
    docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in docs]



from typing import Callable


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )
    
def main() -> None:
    _ = load_dotenv(find_dotenv())

    ALL_TOOLS = qa_tools_for_else(embeddings)

    docs = [
        Document(page_content=t.description, metadata={"index": i})
            for i, t in enumerate(ALL_TOOLS)]
    # print(docs) 

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    def get_tools(query):
        docs = retriever.get_relevant_documents(query)
        return [ALL_TOOLS[d.metadata["index"]] for d in docs]

    

    if user_input := input("Enter your Questions: "):

        if ALL_TOOLS:

            prompt = CustomPromptTemplate(
                template=TEMPLATE,
                tools_getter = get_tools,
                # tools_getter=get_toolsdesc(user_input,retriever,ALL_TOOLS),
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=["input", "intermediate_steps"],
            )
            output_parser = CustomOutputParser()
        else:
            prompt = user_input


        llm = OpenAI(temperature=0)
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        tools = get_toolsdesc(user_input,retriever,ALL_TOOLS)
        tool_names = [tool.name for tool in tools]
        print(tool_names)


        from langchain.memory import ConversationBufferMemory
        memory = ConversationBufferMemory(memory_key="chat_history")

        conversational_agent = initialize_agent(agent='conversational-react-description', 
            tools=tools, 
            llm=llm,    verbose=True,    max_iterations=3,
            # memory=memory,
            return_intermediate_steps=True,
            memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key="output",return_messages=True),
        )
        answer = conversational_agent(user_input)
        print("\n_____",answer.keys())   
        print("\n\n____Output_____\n\n",answer)
        print("\n\n____Steps_____\n\n",answer["intermediate_steps"])
        print("\n\n____Answer_____\n\n",answer["output"])
        
        
if __name__ == "__main__":
    main()
