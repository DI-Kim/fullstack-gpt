import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
import json
import os

os.makedirs(f"./.cache/quiz_files/", exist_ok=True)

st.set_page_config(page_icon="❓", page_title="FullstackGPT | QuizGPT")
st.title("QuizGPT")


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


if "api" not in st.session_state:
    st.session_state["api"] = ""

if not st.session_state["api"]:
    with st.sidebar:
        st.session_state["api"] = st.text_input("Write your openAI API Key.")


else:
    with st.sidebar:
        st.write(st.session_state["api"])
        st.write("repo: https://github.com/DI-Kim/fullstack-gpt")
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=st.session_state["api"],
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(function_call={"name": "create_quiz"}, functions=[function])


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty="EASY"):
    prompt = PromptTemplate.from_template(
        "Make 10 questions about {docs}, shoud follow question's difficulty: {difficulty}"
    )
    chain = prompt | llm
    return chain.invoke({"docs": _docs, "difficulty": difficulty})


@st.cache_data(show_spinner="Searching wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    return retriever.get_relevant_documents(term)


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia Article"))

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file", type=["pdf", "txt", "docx"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:

    def retry():
        st.write("retry")

    if "correct_answer" not in st.session_state:
        st.session_state["correct_answer"] = 0

    len_quiz = 0
    difficulty = st.radio("Select difficulty of questions", ["EASY", "HARD"])
    if "start" not in st.session_state:
        st.session_state["start"] = None
    start = st.button("Generate Quiz")
    if start:
        st.session_state["start"] = "start"

    if st.session_state["start"]:

        response = run_quiz_chain(
            docs, topic if topic else file.name, difficulty
        ).additional_kwargs["function_call"]["arguments"]
        response = json.loads(response)
        # st.write(response["questions"])

        with st.form("questions_form"):
            for idx, question in enumerate(response["questions"]):
                len_quiz = idx + 1
                st.write(question["question"])
                value = st.radio(
                    "Select an option",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"{idx}_radio",
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct✅")
                    st.session_state["correct_answer"] += 1
                elif value is not None:
                    st.error("WrongWrong")
            st.form_submit_button()
        if st.session_state["correct_answer"] < len_quiz:
            st.button("retry", on_click=retry)

        st.write(f"{st.session_state['correct_answer']} / {len_quiz}")
        if st.session_state["correct_answer"] == len_quiz:
            st.balloons()
        st.session_state["correct_answer"] = 0
