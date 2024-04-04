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
import openai
from langchain.memory import ConversationBufferMemory


# 어떤 행동이 일어난 다음 실행되는 말 그대로 callback handler
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    # llm 시작 시 message를 저장할 message_box를 생성한다.
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    # llm 끝날 시, 메세지를 session_state에 저장한다.
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    # 토큰이 생길때마다, 메세지에 토큰을 넣고, 메세지 박스에 넣는다.(출력)
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# 데코레이터: cache에 저장하고 불필요한 재실행을 방지한다. file이 바뀐 걸 파악하여 재실행한다.
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # 올린 파일을 읽고 저장.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    os.makedirs(f"./..cache/files/", exist_ok=True)
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
with st.sidebar:
    if not st.session_state["api"]:
        st.session_state["api"] = st.text_input("Write your openAI API Key.")
        st.write(st.session_state["api"])
    else:
        st.write("Your API key: ", st.session_state["api"])

if st.session_state["api"]:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        api_key=st.session_state["api"],
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
    # 파일을 올렸으면,
    if file:
        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", role="ai", save=False)

        pain_memory_history()
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")

            # 체인 과정
            # 1. prompt에 전달하기 위한 context와 question을 객체에 담는다.
            #    context: retriever로 나온 값을 format_docs 함수에 넣어 하나의 string으로 만듦
            #    question: RunnablePassthrough를 통해 invoke 값 가져옴
            # 2. prompt: 1에서 받은 객체를 prompt에 넣음
            # 3. prompt를 llm모델에 넣어 최종 결과 도출

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

    # 파일을 올리지 않거나 내렸으면,
    else:
        st.session_state["messages"] = []
