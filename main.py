import os 
import streamlit as st
import pickle
import time
import langchain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ.get("OPENAI_API_KEY")

llm = OpenAI(temperature=0.7, max_tokens = 500)
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n','\n','.',','],
    chunk_size =1000,
    chunk_overlap =200 , 
)
embeddings = OpenAIEmbeddings()


st.title("News Research tool 💵📊📈📢")
st.sidebar.title("News Articale URLS")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLS")

file_path = "./vectorindex.pkl"
main_placefolder = st.empty()
if process_url_clicked:
    loaders = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading....Started...✅✅✅✅")
    data = loaders.load()
    main_placefolder.text("Text Splitter.....Started....✅✅✅✅")
    docs = text_splitter.split_documents(data)
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vectors Started Building✅✅✅✅")
    time.sleep(2)
    with open(file_path, 'wb') as f:
        pickle.dump(vectorindex_openai, f)

    
    
query = main_placefolder.text_input("Enter your query here")

if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectorindex = pickle.load(f)
    chain = RetrievalQAWithSourcesChain.from_llm(llm = llm , retriever= vectorindex.as_retriever())                    
    result = chain({'question': query}, return_only_outputs=True)
    st.header("Answers")
    st.write(result['answer'])
    st.header("Source")
    source = result['sources']
    if source :
        st.subheader("Source")
        sources = source.split('/n')
        for source in sources:
            st.write(source)