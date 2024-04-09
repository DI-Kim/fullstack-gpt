from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st


#! https://developers.cloudflare.com/sitemap.xml
# /workers-ai/
# /ai-gateway/
# /vectorize/

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)
st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "question": question,
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}"
        for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
        ],
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    st.write(docs)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


if "api" not in st.session_state:
    st.session_state["api"] = ""

if "url" not in st.session_state:
    st.session_state["url"] = ""

if not st.session_state["api"]:
    with st.sidebar:
        st.write("Please write in order")
        st.session_state["url"] = st.text_input(
            "1. Write down a URL", placeholder="https://example.com"
        )
        st.session_state["api"] = st.text_input("2. Write your openAI API Key.")
        if ".xml" not in st.session_state["url"] and "url" not in st.session_state:
            with st.sidebar:
                st.error("Please write down a Sitemap URL.")
        st.button("Accept")

else:
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=st.session_state["api"],
    )
    with st.sidebar:
        st.write(f'URL: {st.session_state["url"]}')
        st.write(f'API-KEY: {st.session_state["api"]}')
        st.write(
            "REPO: https://github.com/DI-Kim/fullstack-gpt/blob/a640c28545316ff5a5b4f43bd6b5e2436d38ff6b/site_app.py"
        )
    retriever = load_website(st.session_state["url"])
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.write(result.content.replace("$", "\$"))
