# Ask PDF

Ask PDF is a simple and interactive app, which allows you to ask questions about multiple PDFs. It leverages deep learning based language models to provide relevant answers to your queries based on the content of the PDFs.

## Installation

You can install all the dependencies running the following command:

```shell
pip install -r requirements.txt
```

## How to Use

Simply run the main python file and open up a Streamlit app in your browser(by default on  `http://localhost:8501`).
```shell
streamlit run app.py
```
In the sidebar, upload your PDF files and press the 'Process' button. Once your documents are processed, the system is ready to answer your queries.

You can ask a question related to the content of the uploaded PDF files in the text input field labeled 'Ask a question about your PDFs:'.The response from the bot will then be displayed back to you in the chatbot.

## Environment Variables

For this application to function correctly, you need to have the following environment variable set:

* HUGGINGFACEHUB_API_TOKEN

You can set this variable in your `.env` file.

## Technologies Used

This project uses several libraries for its functionality, including:

* Streamlit for creating the web interface
* python-dotenv for environment variable management
* PyPDF2 to extract text from PDF files
* langchain and faiss-cpu for conversation chain generation and text retrieval
* HuggingFace hub API for interacting with Hugging Face models
* FlagEmbedding and sentence_transformers for embeddings

## Intended Use

This tool is not intended for commercial purposes and has been created as a personal project for understanding and experimenting with machine learning large language models and conversation chains.

