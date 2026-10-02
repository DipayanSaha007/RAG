from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, OpenAIError
from colorama import Fore
import warnings

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()

# Load the model
client = OpenAI()

# Define a request
try:
    # The moment this line executes, an HTTP request goes out to OpenAI.
    # Charges apply based on the model used and token count.
    # .create takes 2 parameters: model and messages. The model is the name of the model to use, and messages is a list of messages in the conversation.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You will answer questions and provide information in 1 sentence."},
            {"role": "user", "content": "What is the capital of France?"}]
    )
    print(Fore.GREEN + "Response from OpenAI:")
    print(Fore.YELLOW + response.choices[0].message.content)

except RateLimitError as e:
    print("\n[Error] Insufficient Credits or Rate Limit Exceeded:")
    print("Please check your OpenAI billing balance at: https://platform.openai.com/settings/organization/billing/")
    
except APIError as e:
    print(f"\n[Error] OpenAI API returned an error (Status {e.status_code}): {e.message}")

except OpenAIError as e:
    print(f"\n[Error] General OpenAI client error: {e}")

except Exception as e:
    print(f"\n[Unexpected Error]: {e}")