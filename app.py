import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain

from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()


# Streamlit UI
st.title("AI Document Q&A System")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

question = st.text_input(
    "Ask Question From PDF"
)


if uploaded_file is not None:

    # Save PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")

    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(
        documents
    )

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store vectors
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    if question:

        # Similarity Search
        relevant_docs = vectorstore.similarity_search(
            question
        )

        # Groq LLM
        llm = ChatGroq(
            groq_api_key=os.getenv("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"),
            model_name="llama-3.3-70b-versatile"
        )

        # QA Chain
        chain = load_qa_chain(
            llm,
            chain_type="stuff"
        )

        # Generate Answer
        response = chain.run(
            input_documents=relevant_docs,
            question=question
        )

        st.subheader("Answer")

        st.write(response)