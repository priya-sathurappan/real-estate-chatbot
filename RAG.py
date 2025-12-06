from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain

from prompt import PROMPT, EXAMPLE_PROMPT

load_dotenv()
COLLECTION_NAME = "real-estate"
VECTORSTORE_DIRECTORY = Path(__file__).parent / "resources/vectordb"
EMBEDDING_MODEL = 'Alibaba-NLP/gte-large-en-v1.5'

llm = None
vector_Store = None

def initialize_components():
    """
    This function initializes the components required for RAG
    :return: vectordb, retriever, llm
    """
    global llm, vector_Store
    # Initialize model
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile",
                   api_key=os.getenv("GROQ_API_KEY"),
                   temperature=0.2,
                   max_tokens=500)
    ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
    )
    if vector_Store is None:
        vector_Store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=ef,
                persist_directory= str(VECTORSTORE_DIRECTORY),
    )


def process_urls(urls=None):
    """
    This fuction scraps the content in urls and stored in vectordb
    :param urls: list of urls to be processed
    :return: None
    """
    if urls is None:
        urls = []

    yield "Initializing components"
    initialize_components()
    # best-effort reset; wrapper may not expose this method
    try:
        vector_Store.reset_collection()
    except Exception:
        pass

    yield "Processing urls"
    Loaders = UnstructuredURLLoader(urls=urls)
    data = Loaders.load()

    yield "Splitting text"
    text_Spliiter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n","\n"," "],
    )

    yield "Add doc to vector db"
    docs =  text_Spliiter.split_documents(data)
    print("no of chunks: ",len(docs))
    uuid = [str(uuid4()) for _ in range(len(docs))]
    try:
        vector_Store.add_documents(docs , ids=uuid)
    except TypeError:
        vector_Store.add_documents(docs , id=uuid)


def generate_answer(param):
    # ensure components are initialized (llm and vector store)
    if llm is None or vector_Store is None:
        initialize_components()
    if llm is None:
        raise ValueError("LLM initialization failed.")
    if vector_Store is None:
        raise ValueError("Vector store is not initialized. Please run process_urls() first.")

    qa_chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff",
                                          prompt=PROMPT,
                                          document_prompt=EXAMPLE_PROMPT)
    # Pass llm and the combine_documents_chain as explicit keywords to avoid positional clashes
    chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain,
                                                  retriever=vector_Store.as_retriever(),
                                                  reduce_k_below_max_tokens=True, max_tokens_limit=8000,
                                                  return_source_documents=True
                                                 )
    result = chain.invoke({"question": param}, return_only_outputs=True)
    source = result.get("sources", "")
    return result.get('answer', ""), source


if __name__ == "__main__":
    urls = ['http://economictimes.indiatimes.com/markets/expert-view/2026-would-be-better-for-it-stocks-than-2025-pramod-gubbi/articleshow/125799890.cms?from=mdr',
           'https://www.news18.com/business/economy/rbi-repo-rate-cut-reserve-bank-reduces-interest-rates-by-25-bps-to-5-25-9751108.html']
    for status in process_urls(urls):
        print(status)
    # results = vector_Store.similarity_search("What is the repo rate?", k=2)
    answer,source = generate_answer('What is the repo rate')
    print("Answer: ", answer)
    print("Source: ", source)