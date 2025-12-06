import streamlit as st

from RAG import process_urls, generate_answer

st.title("Real estate assistant")

url1 = st.sidebar.text_input("Enter URL 1", "https://www.example.com/property1")
url2 = st.sidebar.text_input("Enter URL 2", "https://www.example.com/property2")
url3 = st.sidebar.text_input("Enter URL 3", "https://www.example.com/property3")

placeholder = st.empty()
process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    url = [url for url in (url1, url2, url3) if url!='']
    if len(url) == 0:
        placeholder.text("Please enter at least one URL.")
    else:
       for status in process_urls(url):
           placeholder.text(status)
query = placeholder.text_input("Questions")
if query:
    try:
        answer , sources= generate_answer(query)
        st.header("Answer")
        st.write(answer)

        if "sources":
            st.subheader("Source")
            for source in sources.split(","):
                st.write(source)
    except RuntimeError as e:
        st.error(f"you must process url: {e}")


