from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

llm = ChatOpenAI(
    temperature=0.1,
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
    # 유저의 질문에 관해 docs의 인덱스마다 답변과 점수를 달아준다.
    answers_chain = answers_prompt | llm
    # return: 객체 안에 answers라는 큰 리스트와 question이 존재하고, answers의 인덱스는 객체로 이루어져있다.
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
    # get_answers를 통해 나온 return 값의 answers를 condensed로 단순화 및 string화 한다..
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}"
        for answer in answers
    )
    # return: choose_chain으로 question과 condensed를 invoke하여, 최종 결과를 도출한다.
    return choose_chain.invoke({"question": question, "answers": condensed})


# html의 header와 footer 및 필요없는 문자를 없애고 string화 하여 return 한다.
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


# splitter를 통해 긴글 자르기, loader를 통해 원하는 url의 데이터만 가져옴
@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        # filter_urls를 통해 원하는 site만 docs에 넣을 수 있음
        # 정규표현식을 이용해서 원하는 url만 가져올 수 있음.
        # ex) ^(.*\/blog\/).* 는 /blog/ 가 url에 존재하면 docs에 추가 |  ?! 는 반대의 의미 (부정)
        url,
        parsing_function=parse_page,
        # filter_urls=[r"^(.*\/blog\/).*"],
        filter_urls=["https://openai.com/blog/data-partnerships"],
    )
    # request 횟수를 조정해(느리게) 웹사이트에서 차단당하는 것을 막을 수 있음.
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    # 임베딩
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    # return: 임베딩 값을 retiever로 변환
    return vector_store.as_retriever()


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


with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        # 물어볼 질문 입력 칸
        query = st.text_input("Ask a question to the website.")
        if query:
            # docs: url의 텍스트를 임베딩하고, retriever를 진행한 데이터
            # question: query와 같음
            # get_answers: {answers: [{}, {}, ...], question: question}
            # choose_answer: invoke(get_answers return value)
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            # $ 표시가 이상하게 나오므로 \$로 치환
            st.write(result.content.replace("$", "\$"))
