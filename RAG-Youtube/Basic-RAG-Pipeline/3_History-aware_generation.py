from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma

load_dotenv()

# Connect to your document database (Chroma) and set up the retriever
persist_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Set up AI model for generating responses
model = ChatOpenAI(model="gpt-4o")

# Store the conversation history as a list of messages
chat_history = []

def ask_question(user_question):
    print(f"\nUser Question: {user_question}")

    ##### Step 1: Make question clear using the conversation history
    if chat_history:
        # Ask AI to make the question standalone
        message = [
            SystemMessage(content="Given the chat history, rephrase the user's question to be standalone and clear."),
        ] + chat_history + [
            HumanMessage(content=f"User Question: {user_question}")
        ]

        result = model.invoke(message)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    ##### Step 2: Retrieve relevant documents based on the search question
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Retrieved {len(docs)} documents for the question.")

    ##### Step 3: Create a prompt that includes the retrieved documents and the user query
    combined_input = f"""Based on the following documents, answer the question: "{user_question}"\n

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. 
    If you cannot find the answer in the documents, respond with "I don't know"."""

    # Create a ChatOpenAI instance for generating the response
    model = ChatOpenAI(model="gpt-4o")

    ##### Step 4: Define the messages for the chat model to get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that provides answers based on the provided documents."),
        HumanMessage(content=combined_input)
    ]

    # Invoke the model with combined input
    result = model.invoke(messages)
    answer = result.content

    ##### Step 5: Remember the conversation history for future context
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    # Display the generated response
    print("\n---- Generated Response ----")
    print("Content Only:", answer)


# Simple Chat loop
def start_chat():
    print("Welcome to the History-Aware RAG Chatbot! Type 'exit' to quit.")
    while True:
        question = input("\nYour Question: ")
        if question.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
        ask_question(question)

if "__main__" == __name__:
    start_chat()