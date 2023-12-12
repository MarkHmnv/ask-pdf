import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from html_templates import css, user_template, bot_template


def get_pdf_text(pdfs):
    return ''.join(
        page.extract_text()
        for pdf in pdfs
        for page in PdfReader(pdf).pages
    )


def get_conversation_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )


def render_message(message, template):
    st.write(template.replace('{{MSG}}', message), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title='Ask PDF', page_icon=':sparkles:')
    st.write(css, unsafe_allow_html=True)
    st.header('Ask for multiple PDFs :sparkles:')

    st.session_state.setdefault('conversation', None)
    st.session_state.setdefault('chat_history', None)

    question = st.text_input('Ask a question about your PDFs:')

    if question:
        response = st.session_state.conversation({'question': question})
        st.session_state.chat_history = response['chat_history']
        user_messages = st.session_state.chat_history[::2]
        bot_messages = st.session_state.chat_history[1::2]

        for user_message, bot_message in zip(user_messages[::-1], bot_messages[::-1]):
            render_message(user_message.content, user_template)
            render_message(bot_message.content, bot_template)

    with st.sidebar:
        st.subheader('Your documents')
        pdfs = st.file_uploader('Upload your PDFs here and press "Process"', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdfs)
                text_chunks = text_splitter.split_text(raw_text)
                vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    load_dotenv()

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    llm = HuggingFaceHub(repo_id='google/flan-t5-base', model_kwargs={'temperature': 0.5, 'max_length': 512})

    main()
