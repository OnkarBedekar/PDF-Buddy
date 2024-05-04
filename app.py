import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Set environment variables (not necessary if using local API)
os.environ["OPENAI_API_KEY"] = "Not-Needed"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

VectorStore = None

def load_sidebar():
    """Render sidebar with app information."""
    st.sidebar.title('PDF Buddy: Your Friendly Document Assistant!')
    st.sidebar.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [LM Studio](https://lmstudio.ai/)
    ''')
    st.sidebar.write('Made By [Onkar Bedekar](https://www.linkedin.com/in/onkarbedekar/)')

def remove_previous_embeddings():
    """Remove previous embeddings from the 'embeddings' folder."""
    folder = 'embeddings'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            st.write(f"Error while deleting file {file_path}: {e}")

def process_pdf(uploaded_pdf):
    """Process the uploaded PDF file and split it into text chunks."""
    pdf_reader = PdfReader(uploaded_pdf)
    pdf_text = ''.join(page.extract_text() for page in pdf_reader.pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text=pdf_text)

def initialize_llm():
    """Initialize the LLM chatbot."""
    return ChatOpenAI(
        temperature=0.2,
        model_name="Mistral",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_base='http://localhost:1234/v1'
    )

def initialize_vector_store(generated_chunks, uploaded_pdf):
    """Generate and store embeddings for the text chunks."""
    global VectorStore
    store_name = uploaded_pdf.name[:-4]
    embeddings = GPT4AllEmbeddings()
    VectorStore = FAISS.from_texts(generated_chunks, embedding=embeddings)
    with open(f"embeddings/{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

def main():
    load_sidebar()

    # Main app header
    st.header('Chat with PDF')

    # File upload section
    uploaded_pdf = st.file_uploader('Upload your PDF file', type='PDF')
    if uploaded_pdf is not None:
        remove_previous_embeddings()
        generated_chunks = process_pdf(uploaded_pdf)
        initialize_vector_store(generated_chunks, uploaded_pdf)

        llm = initialize_llm()

        if 'messages' not in st.session_state or uploaded_pdf.name != st.session_state.get('previous_file', None):
            st.session_state.messages = []
            st.session_state['previous_file'] = uploaded_pdf.name

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_question := st.chat_input("Ask your question here"):
            with st.chat_message("user", avatar='👨‍🦰'):
                st.markdown(user_question)

            docs = VectorStore.similarity_search(query=user_question + 'and give concise answer', k=3)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.chat_message("assistant", avatar='🤖'):
                with st.spinner('Thinking...'):
                    response = chain.run(input_documents=docs, question=user_question)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)

if __name__ == '__main__':
    main()
