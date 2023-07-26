from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import json
from render import user_msg_container_html_template, bot_msg_container_html_template
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import re
import time

def submit():
    st.session_state.input = st.session_state.widget
    st.session_state.widget = ''

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

    
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF",initial_sidebar_state="expanded")
    st.header("Ask your PDFs💬")

    # Add custom CSS styles to change the background color
    st.markdown(
        """
        <style>
        .main {
            background-color: #98D7C2;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    # upload file
    pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)
            
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'input' not in st.session_state:
        st.session_state.input = ''
            
    user_question = st.text_input("Ask your question and click on the LLM model you want to use:",key='widget', on_change=submit)

    button_clicked_1 = st.button("gpt-4")
    button_clicked_2 = st.button("gpt-3.5-turbo")
    button_clicked_3 = st.button("text-davinci-003")
    
    # extract the text
    if button_clicked_1:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
                # split into chunks
                text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
                chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        #vectorstore = Chroma.from_documents(chunks, embeddings)
        knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

            
        docs = knowledge_base.similarity_search(st.session_state.input)

        
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.1, model="gpt-4"),
            knowledge_base.as_retriever()
        )
        with get_openai_callback() as cb:
          response = qa({"question": st.session_state.input, "chat_history": st.session_state.chat_history})
          st.session_state.chat_history.append((st.session_state.input, response["answer"]))
          
    elif button_clicked_2:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
                # split into chunks
                text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
                chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        #vectorstore = Chroma.from_documents(chunks, embeddings)
        knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

            
        docs = knowledge_base.similarity_search(st.session_state.input)
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo"),
            knowledge_base.as_retriever()
        )
        with get_openai_callback() as cb:
          response = qa({"question": st.session_state.input, "chat_history": st.session_state.chat_history})
          st.session_state.chat_history.append((st.session_state.input, response["answer"]))
          
    elif button_clicked_3:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
                # split into chunks
                text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
                chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        #vectorstore = Chroma.from_documents(chunks, embeddings)
        knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

            
        docs = knowledge_base.similarity_search(st.session_state.input)
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.1, model="text-davinci-003"),
            knowledge_base.as_retriever()
        )
        with get_openai_callback() as cb:
          response = qa({"question": st.session_state.input, "chat_history": st.session_state.chat_history})
          st.session_state.chat_history.append((st.session_state.input, response["answer"]))

    # Display chat history
    for message in st.session_state.chat_history[::-1]:
        if message[0]:
            st.write(user_msg_container_html_template.replace("$MSG", message[0]), unsafe_allow_html=True)
        if message[1]:
            st.write(bot_msg_container_html_template.replace("$MSG", message[1]), unsafe_allow_html=True)
            


                    
if __name__ == '__main__':
    main()

