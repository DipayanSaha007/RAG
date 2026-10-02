from dotenv import load_dotenv
from langchain.chains import RetrievalQA

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from colorama import Fore
import os
import warnings 

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()

# Load the model
model = ChatOpenAI()


# Load the documents = https://python.langchain.com/docs/get_started/introduction
loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
documents = loader.load()

# prompt templates
template = "you are senior developer who answers {question}  based on your knowledge based on {context}"
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

# Creating the RAG Pipeline
# RETRIEVER - Load embeddings and create a vector store 
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)


# GENERATE - Define the function to generate the response
def generate(query): 
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 1}), chain_type_kwargs=chain_type_kwargs,)
    response = chain.run(query)
    return response



def start():
    print("OPENAI API KEY", os.getenv("OPENAI_API_KEY"))
    print("documents", documents)
    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:
            response = generate(user_input)

            print(Fore.BLUE + f"A: " + response + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()