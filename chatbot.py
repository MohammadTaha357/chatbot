import streamlit as st

import os
from langchain_groq import ChatGroq # groq llm integration
from langchain_classic.memory import ConversationBufferMemory # memory backend for chat
from langchain_classic.chains import ConversationChain # it wires LLM + memory

## Streamlit
st.set_page_config(page_title=" 💬 Conversational Chatbot") # title in browser tab 

st.title("💬 Conversational Chatbot with Message History") # app header

# sidebar control
key=st.sidebar.text_input("Enter api key")
model_name = st.sidebar.selectbox( 
    "Select Groq Model",
    ["llama-3.3-70b-versatile ","moonshotai/kimi-k2-instruct-0905","openai/gpt-oss-120b"]
    )
                                                                                                                
temperature = st.sidebar.slider( # fix the randomness of the response
    "Temperture",0.0,1.0,0.7
)

max_tokens = st.sidebar.slider( # max response length
    "Max Tokens", 50,300,150
)
clear=st.sidebar.button("Clear Chat")
if "memory" not in st.session_state:
    # perssist memory across reruns
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True # return as list of memory, not in one big string.
    )

if "history" not in st.session_state:
    st.session_state.history=[]

# user input

user_input = st.chat_input("Talk to an amazing chatbot ") # clears itself on enter

if user_input:
    st.session_state.history.append(("user", user_input))

    # initialized a fresh llm for this turn
    llm= ChatGroq(
        api_key=key,
        model=model_name,
        temperature= temperature,
        max_tokens= max_tokens
    )

    # build conversation chain in our memory
    conv =  ConversationChain(
        llm=llm,
        memory= st.session_state.memory,
        verbose= True
    )

    ## get ai response ( memory is updted internally)
    ai_response = conv.predict(input=user_input)

    # append assitant to history
    st.session_state.history.append(("assistant",ai_response))

# render  chat bubble
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text) # user style
    else:
        st.chat_message("assistant").write(text) # assistant style

if clear:
    st.session_state.history=[]




