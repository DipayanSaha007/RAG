from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

#1. Load the vector store from the specified directory
persist_directory = "db/chroma_db"
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_metadata={"hnsw:space": "cosine"}  ## Comparison metric for the vector store, can be "cosine", "l2", or "ip" (inner product)
)

#2. Search for relevant documents based on a query
query = "Which island does SpaceX lease for its launches in the Pacific?"

retriever = db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant documents

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display Results
print("---- Context ----")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}")


### 3. Use the retrieved documents to generate a response using a language model
# Create a prompt that includes the retrieved documents and the user query
combined_input = f"""Based on the following documents, answer the question: "{query}"\n

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. 
If you cannot find the answer in the documents, respond with "I don't know"."""

# Create a ChatOpenAI instance for generating the response
model = ChatOpenAI(model="gpt-4o")

# Define the messages for the chat model
messages = [
    SystemMessage(content="You are a helpful assistant that provides answers based on the provided documents."),
    HumanMessage(content=combined_input)
]

# Invoke the model with combined input
result = model.invoke(messages)

# Display the generated response
print("\n---- Generated Response ----")
# print("Full Result:", result)
print("Content Only:", result.content)