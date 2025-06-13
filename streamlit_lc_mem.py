from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
import random
import streamlit as st
from langchain_core.messages import HumanMessage
import numpy as np

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langsmith_key = os.getenv('LANGSMITH_API_KEY')


model = init_chat_model(
    model = 'gpt-4o-mini',
    temperature = 0,
    max_tokens = 1028,
)


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
st.title("A Simple LangChain, Streamlit Chatbot with Session State Memory")
st.write("This is a simple chatbot that can talk to you. \
         It uses the GPT-4o-mini model to generate response")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

def response_chunk_by_chunk(prompt):
    for chunk in model.stream(prompt):
        yield chunk.content

if prompt := st.chat_input("Say something"):
    # messages = st.container(height=300)
    st.session_state.messages.append({
        'role' : 'user',
        'content': prompt,
    })
    with st.chat_message("user"):
        st.write(prompt)
    # messages.chat_message("assistant").write(model.invoke(prompt).content)
    with st.chat_message("assistant"):
        response = st.write_stream(response_chunk_by_chunk(str(st.session_state.messages) + prompt))
        st.session_state.messages.append({
            'role' : 'assitant',
            'content' : response,
        })
