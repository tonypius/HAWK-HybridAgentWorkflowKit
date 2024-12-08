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
from PyPDF2 import PdfReader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
import streamlit as st
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

# initialize the models
from langchain.chat_models import AzureChatOpenAI

os.environ["AZURE_OPENAI_KEY"] = "secret-key-here"
os.environ["AZURE_OPENAI_ENDPOINT"] = "endpoint-here"
verbose = False

BASE_URL = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT_NAME = 'deployment-name-here' #This will correspond to the custom name you chose for your deployment when you deployed a model.
OPENAPI_VERSION ="2023-05-15"
openai = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=OPENAPI_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)
# openai.temperature = 0.1

global_file_name = None
global_file_extension = None


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


def init_page() -> None:
    st.set_page_config(
        page_title="KFH GPT"
    )
    st.sidebar.title("Options")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content=(
                    "You are a helpful AI QA assistant. "
                    "When answering questions, use the context enclosed by triple backquotes if it is relevant. "
                    "If you don't know the answer, just say that you don't know, "
                    "don't try to make up an answer. "
                    "Reply your answer in mardkown format.")
            )
        ]
        st.session_state.costs = []


def get_pdf_text() -> Optional[str]:
    """
    Function to load PDF text and split it into chunks.
    """
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        label="Here, upload your PDF file you want ChatGPT to use to answer",
        type=["csv", "txt", "pdf"]
    )
    try:
        file_name = (uploaded_file.name).split(".")
        global global_file_name 
        global_file_name = file_name[0]
        global global_file_extension 
        global_file_extension = uploaded_file.name
    except:
        pass

    if uploaded_file:
        print("FILE NAME__",uploaded_file.name)
        file = uploaded_file.name
        # docs = None

        if file.endswith('.txt'):
            print("text")
            docume = [uploaded_file.read().decode()]
            # documents = text.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents(docume)
            return docs
        
        elif file.endswith('.csv'):
            print("csv")
            import pandas as pd
            encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
            df = pd.read_csv(uploaded_file, encoding="latin1")

            _path = r"temp.csv"
            if _path:
                df.to_csv(_path, index=False)

            docume = CSVLoader(_path,encoding='latin1')
            documents = docume.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            docs = text_splitter.split_documents(documents)
            return docs
        
        elif file.endswith('.pdf'):
            print("pdf")
            pdf_reader = PdfReader(uploaded_file)
            text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            docs = text_splitter.split_text(text)
            return docs
    else:
        return None


def build_vectore_store(
    texts: str, embeddings: Union[OpenAIEmbeddings, LlamaCppEmbeddings, SentenceTransformerEmbeddings]) \
        -> Optional[Chroma]:
    """
    Store the embedding vectors of text chunks into vector store (chroma).
    """

    if texts:
        print("_________LOADING USER DATA__________")
        with st.spinner("Loading USER DATA ...!!"):
            print(" loading")

            print(global_file_extension)

            if global_file_extension.endswith('.pdf'):
                print("pdf to chroma")
                chroma = Chroma.from_texts(texts, embeddings,persist_directory='./user_pdf')
            else:
                print("csv,txt to chroma")
                chroma = Chroma.from_documents(texts, embeddings,persist_directory='./user_pdf')


        tools = qa_tools_for_if(embeddings)

        st.success("File Loaded Successfully!!")

    else:
        # qdrant = None
        print("_________LOADING TEXT INPUT_________")
        st.spinner("Loading KFH data ...!!")
        tools = qa_tools_for_else(embeddings)

    return tools

def qa_tools_for_if(embeddings):

    retriever_user = Chroma(persist_directory='./user_pdf',embedding_function=embeddings)
    retriever_user.persist()
    user_data = RetrievalQA.from_chain_type(
        llm=openai, chain_type="stuff", retriever=retriever_user.as_retriever()
    )
    retriever_bank = Chroma(persist_directory='./bank',embedding_function=embeddings)
    retriever_bank.persist()
    bank_data = RetrievalQA.from_chain_type(
        llm=openai, chain_type="stuff", retriever=retriever_bank.as_retriever()
    )
    retriever_esg = Chroma(persist_directory='./esg',embedding_function=embeddings)
    retriever_esg.persist()
    esg_data = RetrievalQA.from_chain_type(
        llm=openai, chain_type="stuff", retriever=retriever_esg.as_retriever()
    )

    # print(global_file_name)

    tools = [
        Tool(
            name=global_file_name,
            func=user_data.run,
            description="useful for when you need to answer questions if the question contains the keyword from the given or uploaded documents, not related to ESG and Bank. Input should be a fully formed question.",
        ),
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

def qa_tools_for_else(embeddings):

    retriever_bank = Chroma(persist_directory='./bank',embedding_function=embeddings)
    retriever_bank.persist()
    bank_data = RetrievalQA.from_chain_type(
        llm=openai, chain_type="stuff", retriever=retriever_bank.as_retriever()
    )

    retriever_esg = Chroma(persist_directory='./esg',embedding_function=embeddings)
    retriever_esg.persist()
    esg_data = RetrievalQA.from_chain_type(
        llm=openai, chain_type="stuff", retriever=retriever_esg.as_retriever()
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


def select_llm() -> Union[ChatOpenAI, LlamaCpp, HuggingFaceTextGenInference]:
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("llama-2-13b-GPTQ",
                                   "llama-2-13b-hf-chat",
                                   "falcon-7b",
                                   "gpt-3.5-turbo-0613",
                                   "gpt-3.5-turbo-16k-0613",
                                   "gpt-4"))
    temperature = 0.01
    # temperature = st.sidebar.slider("Temperature:", min_value=0.0,
    #                                 max_value=1.0, value=0.0, step=0.01)
    return model_name, temperature


def load_llm(model_name: str, temperature: float) -> Union[ChatOpenAI, LlamaCpp, HuggingFaceTextGenInference]:
    """
    Load LLM.
    """
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=temperature, model_name=model_name)
    elif model_name.startswith("llama-2-13b-GPTQ"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return HuggingFaceTextGenInference(
            inference_server_url="llm-domain-here",
            max_new_tokens=1512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.5,
            repetition_penalty=1.03,
        )
    elif model_name.startswith("llama-2-13b-hf-chat"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return HuggingFaceTextGenInference(
            inference_server_url="llm-domain-here",
            max_new_tokens=1512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
        )

def load_embeddings(model_name: str) -> Union[OpenAIEmbeddings, LlamaCppEmbeddings, SentenceTransformerEmbeddings]:
    """
    Load embedding model.
    """
    if model_name.startswith("gpt-"):
        # return OpenAIEmbeddings()
        return SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en") #BAAI/bge-small-en thenlper/gte-large
    elif model_name.startswith("llama-2-"):
        #return LlamaCppEmbeddings(model_path=f"./models/{model_name}.bin")
        return SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")


def get_answer(llm, messages) -> tuple[str, float]:
    """
    Get the AI answer to user questions.
    """
    if isinstance(llm, ChatOpenAI):
        with get_openai_callback() as cb:
            answer = llm(messages)
        return answer.content, cb.total_cost
        #return answer.content
    if isinstance(llm, LlamaCpp):
        #return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0
        return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0
    if isinstance(llm, HuggingFaceTextGenInference):
        return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0

def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant
    format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    # DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. If you don't know the answer to a question, please don't share false information. When answering questions, use the context enclosed by triple backquotes if it is relevant. If you don't know the answer, just say that you don't know, don't try to make up an answer. Reply your answer in mardkown format. """

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def extract_userquesion_part_only(content):
    """
    Function to extract only the user question part from the entire question
    content combining user question and pdf context.
    """
    content_split = content.split("[][][][]")
    if len(content_split) == 3:
        return content_split[1]
    return content


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

    init_page()

    model_name, temperature = select_llm()
    llm = load_llm(model_name, temperature)
    embeddings = load_embeddings(model_name)

    texts = get_pdf_text()
    # qdrant = build_vectore_store(texts, embeddings)
    ALL_TOOLS = build_vectore_store(texts, embeddings)

    # print("_____Generated Tools___:",ALL_TOOLS)

    # tool_names = [tool.name for tool in ALL_TOOLS]
    # print(tool_names)

    docs = [
        Document(page_content=t.description, metadata={"index": i})
            for i, t in enumerate(ALL_TOOLS)]
    # print(docs) 

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    def get_tools(query):
        docs = retriever.get_relevant_documents(query)
        return [ALL_TOOLS[d.metadata["index"]] for d in docs]

    init_messages()

    # Supervise user input
    if user_input := st.chat_input("How may I help you?"):
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

        with st.spinner("Personal LLM typing ..."):

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
                memory=memory,
            )


            answer = conversational_agent(user_input)
            print(answer["output"])

        st.session_state.messages.append(AIMessage(content=(user_input + "\n\n" + answer["output"])))

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(extract_userquesion_part_only(message.content))

    costs = st.session_state.get("costs", [])


# streamlit run app.py
if __name__ == "__main__":
    main()
