import numpy as np
import faiss
import json

from configparser import ConfigParser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


def setup_client():
  """Set up the Mistral AI client

  Arguments:
      private_key: The private key to make for the account making the transactions

  Returns:
      client: The client variable that connects to either the Testnet or Mainnet
  """
  config = ConfigParser()
  config.read('config_mistral.ini')
  private_key = config.get('Account_Details', 'private_key')
  client = MistralClient(api_key=private_key)
  return client


def get_text_embedding(input, client):
    embeddings_batch_response = client.embeddings(
          model="mistral-embed",
          input=input
    )
    return embeddings_batch_response.data[0].embedding

def run_mistral(user_message, client, model):
    messages = [
        ChatMessage(role="user", content=user_message)
    ]
    chat_response = client.chat(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)


if __name__ == "__main__":

  client = setup_client()
  
  # Open food menu
  file = open("menu.json", "r")
  content = file.read()
  #print(content)
  file.close()

  with open('menu.json') as menu:
    menu_dict = json.load(menu)
    menu_string = json.dumps(menu_dict) 

  text = menu_string

  # Separate the text into chunks
  chunk_size = 2048
  chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

  # Create the text embeddings:
  text_embeddings = np.array([get_text_embedding(chunk, client) for chunk in chunks])

  # Save into vector database, Faiss is just an example
  d = text_embeddings.shape[1]
  index = faiss.IndexFlatL2(d)
  index.add(text_embeddings)

  question = input("What would you like to order? ")
  question_embeddings = np.array([get_text_embedding(question, client)])

  D, I = index.search(question_embeddings, k=2) 
  retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

  prompt = f"""
  Context information is below.
  ---------------------
  {retrieved_chunk}
  ---------------------
  Given the context information and not prior knowledge, answer the query.
  Query: {question} + "use natural language and keep your answers short and concise"
  Answer:
  """
  
  # Select the model and run Mistral
  result = run_mistral(prompt, client, model="mistral-medium-latest")
  
  print(result)