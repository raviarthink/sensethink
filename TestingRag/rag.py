import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# Load environment variables
load_dotenv()

# client = OpenAI(api_key=api_key)

app = Flask(__name__)


# Global variable to hold the vector database
vectordb = None

def read_and_split_data(directory='data'):

    
    # Load the document
    loader = DirectoryLoader(directory)
    docs = loader.load()
    
    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    
    return splits

def create_embeddings_and_store(splits, persist_directory='docs/chroma/'):
    # Embeddings
    embedding = OpenAIEmbeddings()
    
    # Initialize the vector store
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    
    return vectordb

def query_data(vectordb, query_text):
    # Load environment variables
    
    llm_name = 'gpt-3.5-turbo-0125'
    
    # Initialize the LLM
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    
    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    
    # Test the retrieval
    result = qa_chain({"query": query_text})
    
    return result["result"]

@app.route('/incred', methods=['POST'])
def handle_query():
    global vectordb
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({"error": "Query text is required."}), 400
    
    # Initialize vector database if it doesn't exist
    if vectordb is None:
        if not os.path.exists('docs/chroma/'):
            splits = read_and_split_data()
            vectordb = create_embeddings_and_store(splits)
        else:
            vectordb = Chroma(
                persist_directory='docs/chroma/',
                embedding_function=OpenAIEmbeddings()
            )
    
    # Query the vector database
    result = query_data(vectordb, query_text)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)