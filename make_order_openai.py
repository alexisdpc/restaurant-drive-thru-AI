import json

from time import sleep
from openai import OpenAI
from configparser import ConfigParser

def read_credentials():
  """Read the account and private key from the config.ini file

  Returns:
      list with two elements: the account number and the private key.
  """

  config = ConfigParser()
  config.read('config.ini')

  private_key = config.get('Account_Details', 'private_key')

  return private_key


def setup_client(private_key):
  """Set up the OpenAI client

  Arguments:
      private_key: The private key to make for the account making the transactions

  Returns:
      client: The client variable that connects to either the Testnet or Mainnet
  """
  #client = OpenAI( api_key = os.environ.get("OPENAI_API_KEY"))
  client = OpenAI(api_key = private_key)
  return client

def create_assistant(client):  

  # Select the RAG file
  file = client.files.create(
    file = open("menu.json", "rb"),
    purpose = 'assistants'
  )
  
  # Create an OpenAI assistant
  assistant = client.beta.assistants.create(
    name = 'restaurant drive-thru',
    instructions = "You are a restaurant, use your knowledge to take the orders input by the client." ,
    model = "gpt-4-turbo-preview", #"gpt-3.5-turbo", #"gpt-4-1106-preview", # "gpt-3.5-turbo", #"gpt-4-turbo-preview"
    tools =[{"type": "retrieval"}],
    file_ids = [file.id]
  )

  return [file, assistant]


def create_thread(client):
  thread = client.beta.threads.create()
  return thread


def make_an_order(client, file, assistant, thread):
  
  # User inputs the order
  user_input = input("What would you like to order? ")

  messages = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = user_input + "be short and concise in your answer.",
    file_ids = [file.id],
  )

  run = client.beta.threads.runs.create(
    thread_id = thread.id,
    assistant_id = assistant.id,
    instructions = "The user has a premium account."
  )

  #print(run.status)

  # Run the client until "completed"
  while (run.status != "completed"):
  
    #print("Waiting for the Agent to respond...")
    run = client.beta.threads.runs.retrieve(
      thread_id = thread.id,
      run_id = run.id
    )
    #print (run.status)

  messages = client.beta.threads.messages.list(
    thread_id = thread.id
  )  

  #for m in messages:
    #print(m.role + ":" + str(m.content[0].text))

  return messages 

if __name__ == "__main__":
  
  private_key = read_credentials() # Obtain the private key
  client = setup_client(private_key) # Setup the OpenAI client
  [file, assistant] = create_assistant(client) # Select the file and create assistant
  thread = create_thread(client) # Create the thread

  # Run this to make an order :)
  output = make_an_order(client, file, assistant, thread) 


  # Save the output to a file
  result = output.json()
  output_json = json.loads(result)

  num_outputs = len(output_json['data'])

  for i in range(num_outputs-1):
    print(output_json['data'][num_outputs-2-i]['content'][0]['text']['value'] + '\n') 

  with open('output.txt', 'w') as f:
    for i in range(num_outputs):
      f.write(output_json['data'][num_outputs-1-i]['content'][0]['text']['value'] + '\n') 