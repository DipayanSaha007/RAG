import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#1. Load documents from the specified directory
def load_documents(docs_path):
    """Load all text files from the specified directory"""

    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified directory '{docs_path}' does not exist.")

    # Load all text files from the directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    if len(documents) == 0:
        raise ValueError(f"No text files found in the directory '{docs_path}'.")

    return documents

#2. Split the documents into smaller chunks
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks for processing"""

    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    if len(chunks) == 0:
        raise ValueError("No chunks were created from the documents.")

    return chunks

#3. Create embeddings for the chunks
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create a vector store from the document chunks"""

    print("Creating embeddings for the chunks...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create ChromaDB vector store from the document chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}    ## Comparison metric for the vector store, can be "cosine", "l2", or "ip" (inner product)
    )
    vector_store.persist()
    print(f"Vector store created and persisted at '{persist_directory}'.")
    return vector_store

# The Main function to run the ingestion pipeline
def main():
    print("Starting the ingestion pipeline...")

    #1. Load documents from the specified directory
    documents = load_documents(docs_path = "docs")
    #2. Split the documents into smaller chunks
    chunks = split_documents(documents)
    #3. Create embeddings for the chunks
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()