from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv 
from langchain_core.messages import HumanMessage


load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langsmith_key = os.getenv('LANGSMITH_API_KEY')


model = init_chat_model(
    model = 'gpt-4o-mini',
    temperature = 0,
    max_tokens = 1024,
    model_provider='openai',
)


def get_model_response(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages' : response}

workflow_builder = StateGraph(MessagesState)
workflow_builder.add_node('get_model_response', get_model_response)
workflow_builder.add_edge(START, 'get_model_response')
memory = MemorySaver()
workflow = workflow_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

loop = True
while(loop):
    prompt = input('Enter your query: ')
    if(prompt.lower() == 'bye'):
        loop = False
    else:
        input_msg = [HumanMessage(content = prompt)]
        response = workflow.invoke({'messages' : input_msg}, config = config)
        response['messages'][-1].pretty_print()
