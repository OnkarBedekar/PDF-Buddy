# PDF Buddy

PDF Buddy is a Python chatbot developed using LangChain for interacting with PDF files. It integrates a locally running Large Language Model (LLM) using LM Studio for natural language processing. The chatbot creates embeddings of PDF files for enhanced document understanding and implements a vector database for efficient storage and retrieval of embeddings.


## Features

- PDF Interaction: The chatbot allows users to interact with PDF files, extracting information, answering questions, and performing actions based on the content.
- Natural Language Processing: Integrated with a Large Language Model (LLM) for processing natural language queries and commands.
- Embedding Generation: Generates embeddings for PDF files, enabling better understanding and context-based responses.
- Vector Database: Implements a vector database for storing and retrieving embeddings, ensuring efficient data access.
- LLM: Utilizes Mistral Instruct V0 2 7B parameter LLM for enhanced language processing capabilities.
- Streamlit UI: Uses Streamlit for creating a user-friendly interface, enabling seamless interaction with the chatbot and PDF files.

## Usage

- Clone the Repository and Install Dependencies
```bash
    git clone https://github.com/OnkarBedekar/PDF-Buddy.git
    cd pdf-buddy
    pip install -r requirements.txt
```

- Install LM Studio and Add 'Mistral Model'
    - Follow the installation instructions for LM Studio.
    - Add the 'Mistral Model' or any other model you prefer for natural language processing.

- Run the Application
```bash
    streamlit run app.py
```

- Upload a PDF File
    - Click on the file upload button and select a PDF file from your computer.
    - Wait for the input box to appear indicating that the PDF file has been processed.

- Ask Questions on the PDF File
    - Once the input box appears, you can start asking questions or giving commands related to the content of the PDF file.
    - The chatbot will process your input and provide responses based on the PDF content.

- Upload Another PDF File (Optional)
    - If you wish to ask questions on another PDF file, simply upload a new PDF file using the file upload button.
    - The chatbot will switch to processing the new PDF file, and you can start asking questions on the new content.
