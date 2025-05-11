import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.tools import Tool

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Wikipedia and Arxiv tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

# Custom PDF tool
loader = PyPDFLoader("./F1-Reg.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()

def formatted_f1_retriever(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])

retriever_tool = Tool(
    name="Formula1_Regulations",
    func=formatted_f1_retriever,
    description="Useful for answering questions about Formula 1 Sporting Regulations. Input should be a question or topic."
)

# ---------------- UI Starts ----------------

st.set_page_config(page_title="LangChain Chat Search", layout="wide")
st.title("💬 SmartQuery: Chat with Search Tools")

with st.expander("ℹ️ About this app"):
    st.markdown("""
    This chatbot uses LangChain agents and tools to retrieve data from:
    - Wikipedia
    - Arxiv
    - DuckDuckGo
    - A custom PDF on Formula 1 Sporting Regulations

    Simply enter a question below and watch it search multiple sources intelligently.
    """)

# Sidebar setup
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### 🔑 API Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password", placeholder="sk-...")

# Chat memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi, I'm a chatbot that can search the web and documents. Ask me anything!"}
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    st.markdown("---")

# Chat input
if prompt := st.chat_input(placeholder="Ask me something like: What is LangChain?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [wiki, arxiv, search, retriever_tool]

    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)
