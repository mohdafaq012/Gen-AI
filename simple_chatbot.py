from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace(llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",  # Specify the model repository ID
    task="text-generation",  # Specify the task you want to perform
    max_new_token = 10,
    temperature=0.1
))

chat_history = [
    SystemMessage(content = "You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI:", response.content)
    
print("Chat History:", chat_history)  # Print the chat history for reference
