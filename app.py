import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import os
from langchain.memory import ConversationBufferMemory

os.makedirs(f"./..cache/files/", exist_ok=True)
os.makedirs(f"./..cache/embeddings/", exist_ok=True)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):

    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def pain_memory_history():
    for content in memory.load_memory_variables({})["history"]:
        st.write(content)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(input):
    return memory.load_memory_variables({})["history"]


def invoke_chain(question):
    result = chain.invoke(question)
    memory.save_context({"input": question}, {"output": result.content})


memory = ConversationBufferMemory(return_messages=True)

st.set_page_config(page_icon="📃", page_title="DocumentGPT")
st.title("DocumentGPT")

st.session_state["api"] = ""
if not st.session_state["api"]:
    with st.sidebar:
        st.session_state["api"] = st.text_input("Write your openAI API Key.")
        st.write(st.session_state["api"])


if st.session_state["api"]:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=st.session_state["api"],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful assistant. Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

  Context: {context}
""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    st.markdown(
        """
  WELCOME
              
              Use this chatbot to ask questions to an AI about your files!
  """
    )
    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
        )
        st.write("repo: https://github.com/DI-Kim/fullstack-gpt")

    if file:
        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", role="ai", save=False)

        pain_memory_history()
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")

            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnablePassthrough.assign(history=load_memory)
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                invoke_chain(message)

    else:
        st.session_state["messages"] = []
