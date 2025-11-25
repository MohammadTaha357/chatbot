import streamlit as st
from pydantic import SecretStr

import os
from langchain_groq import ChatGroq # groq llm integration
from langchain_classic.memory import ConversationBufferMemory # memory backend for chat
from langchain_classic.chains import ConversationChain # it wires LLM + memory

## Streamlit
st.set_page_config(page_title=" 💬 Conversational Chatbot") # title in browser tab 

st.title("💬Conversational Chatbot with Message History") # app header

# sidebar control
st.sidebar.markdown("## Configuration")
key=st.sidebar.text_input("Enter Groq API Key",type="password")
user_name = st.sidebar.text_input("Enter your name (optional)")

st.sidebar.markdown("### Model Selection")
model_name = st.sidebar.selectbox( 
    "Select Groq Model",
    ["openai/gpt-oss-120b","moonshotai/kimi-k2-instruct-0905"]
    )

st.sidebar.markdown("### Generation Parameters")
temperature = st.sidebar.slider( # fix the randomness of the response
    "Temperature",0.0,1.0,0.7
)




# system_prompt = st.sidebar.text_area("Enter a custom system prompt (optional)", height=100)

st.sidebar.markdown("### Chat Controls")
clear=st.sidebar.button("Clear Chat")
reset_session=st.sidebar.button("Reset Session (Clear Memory & History)")

show_memory = st.sidebar.checkbox("Show Conversation Memory", value=False)

import json
from io import StringIO

if "memory" not in st.session_state:
    # perssist memory across reruns
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True # return as list of memory, not in one big string.
    )

if "history" not in st.session_state:
    st.session_state.history=[]

    # Add download chat history button and logic
if st.session_state.history:
        
        # Prepare chat history as a formatted text string
        chat_history_text = ""
        for msg in st.session_state.history:
            if len(msg) == 3:
                role, text, timestamp = msg
            elif len(msg) == 2:
                role, text = msg
                timestamp = ""
            else:
                continue
            chat_history_text += f"{role.capitalize()}: {text} ({timestamp})\n"
    # Download button for chat history
        st.sidebar.download_button(
        label="Download Chat History",
        data=chat_history_text,
        file_name="chat_history.txt",
        mime="text/plain"
        )
if reset_session:
    st.session_state.history=[]
    if "memory" in st.session_state:
        del st.session_state.memory
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True
        )

# user input

user_input = st.chat_input("Talk to an amazing chatbot ") # clears itself on enter


import datetime


if user_input:

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(("user", user_input, timestamp))

    with st.spinner("Generating response..."):
        # initialized a fresh llm for this turn
        llm= ChatGroq(
            api_key=SecretStr(key) if key else None,
            model=model_name,
            temperature= temperature
        )

        # build conversation chain in our memory
        # if system_prompt.strip() != "":
        #     full_prompt = system_prompt + "\n\n"
        #     # Warning: ConversationChain does not have direct system prompt parameter,
        #     # so we prepend prompt to input
        #     input_with_prompt = full_prompt + user_input
        #     conv =  ConversationChain(
        #         llm=llm,
        #         memory= st.session_state.memory,
        #         verbose= True
        #     )
        #     ai_response = conv.predict(input=input_with_prompt)
        
        conv =  ConversationChain(
            llm=llm,
            memory= st.session_state.memory,
            verbose= True
        )
        ai_response = conv.predict(input=user_input)

    # append assitant to history
    st.session_state.history.append(("assistant",ai_response, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Retry last assistant response button logic
if len(st.session_state.history) >= 2:
    last_user_msg = None
    last_assistant_msg = None
    # Find last user and assistant pair
    for i in range(len(st.session_state.history) - 1, 0, -1):
        if st.session_state.history[i][0] == "assistant" and st.session_state.history[i-1][0] != "assistant":
            last_assistant_msg = st.session_state.history[i]
            last_user_msg = st.session_state.history[i-1]
            break

    if last_user_msg and last_assistant_msg:
        if st.sidebar.button("Retry Last Response"):
            # Remove last assistant message
            st.session_state.history.pop()
            user_input_retry = last_user_msg[1]

            # Re-run LLM with last user input to regenerate response
            llm_retry = ChatGroq(
                api_key=SecretStr(key) if key else None,
                model=model_name,
                temperature= temperature
            )
            conv_retry = ConversationChain(
                llm=llm_retry,
                memory=st.session_state.memory,
                verbose=True
            )
            ai_response_retry = conv_retry.predict(input=user_input_retry)
            st.session_state.history.append(("assistant", ai_response_retry, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# render  chat bubble
for msg in st.session_state.history:
    if len(msg) == 3:
        role, text, timestamp = msg
    if role=="user":
        with st.chat_message("user"):
            st.write(text)
    elif role=="assistant":
        with st.chat_message("assistant"):
            st.write(text)
if show_memory:
    st.sidebar.markdown("### Conversation Memory")
    memory_msgs = st.session_state.memory.load_memory_variables({}).get("history", [])
    if memory_msgs:
        for i, msg in enumerate(memory_msgs):
            st.sidebar.markdown(f"**{i+1}. {msg.type.capitalize()}**: {msg.content}")
    else:
        st.sidebar.info("Memory is empty.")
if clear:
    st.session_state.history=[]



