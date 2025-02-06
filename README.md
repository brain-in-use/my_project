# My Project: Setting Up Ollama & Running DeepSeek R1 Locally for a Powerful RAG System

## Description
This project is designed to help you set up Ollama and run DeepSeek R1 locally, creating a powerful Retrieval-Augmented Generation (RAG) system. It aims to provide a seamless and efficient way to integrate these tools for enhanced data retrieval and generation capabilities, including the ability to upload a PDF and get specific output from queries asked.

## Features
- Easy setup for Ollama
- Local deployment of DeepSeek R1
- Enhanced data retrieval and generation
- Upload PDF and query specific information

## Prerequisites
Before running the RAG system, make sure you have:

- Python installed
- Conda environment (Recommended for package management)
- Required Python packages

## Installation
To install this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/my_project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd my_project
    ```
3. Install the dependencies:
    ```bash
    npm install
    ```

To install the required Python packages, run the following commands:

```bash
pip install -U langchain langchain-community
pip install streamlit
pip install pdfplumber
pip install semantic-chunkers
pip install open-text-embeddings
pip install faiss
pip install ollama
pip install prompt-template
pip install langchain
pip install langchain_experimental
pip install sentence-transformers
pip install faiss-cpu
```

#Running the RAG System
Create a new project directory:
```bash
mkdir rag-system && cd rag-system
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
```
# Streamlit UI
```bash
st.title("📄 RAG System with DeepSeek R1 & Ollama")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = Ollama(model="deepseek-r1:1.5b")

    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        response = qa(user_input)["result"]
        st.write("**Response:**")
        st.write(response)
```
#Step 5: Running the App
Once the script is ready, start your Streamlit app:
```bash
streamlit run app.py
```

Upload a PDF file through the provided interface.
Enter your query to retrieve specific information from the uploaded PDF.
Contributing
If you would like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
If you have any questions, feel free to reach out at braininuse1@gmail.com. ```