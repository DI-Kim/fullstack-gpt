from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


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
        .replace("View all articles ", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(
        # filter_urls를 통해 원하는 site만 docs에 넣을 수 있음
        # 정규표현식을 이용해서 원하는 url만 가져올 수 있음.
        # ex) ^(.*\/blog\/).* 는 /blog/ 가 url에 존재하면 docs에 추가 |  ?! 는 반대의 의미 (부정)
        url,
        filter_urls=[r"^(.*\/blog\/).*"],
        parsing_function=parse_page,
    )
    # request 횟수를 조정해(느리게) 웹사이트에서 차단당하는 것을 막을 수 있음.
    # loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


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
            st.error("Please write down a Sitemap URL")
    else:
        docs = load_website(url)
        st.write(docs)
