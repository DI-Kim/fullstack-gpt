from langchain.document_loaders import SitemapLoader
import streamlit as st


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    loader = SitemapLoader(
        # filter_urls를 통해 원하는 site만 docs에 넣을 수 있음
        url,
        filter_urls=["https://openai.com/blog/data-partnerships"],
    )
    # request 횟수를 조정해(느리게) 웹사이트에서 차단당하는 것을 막을 수 있음.
    # loader.requests_per_second = 1
    docs = loader.load()
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
