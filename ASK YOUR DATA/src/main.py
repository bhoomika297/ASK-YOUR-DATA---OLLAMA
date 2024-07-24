import pandas as pd
import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

# STREAMLIT web App configuration

st.set_page_config(
    page_title="ASK YOUR DATA",
    page_icon="💬",
    layout = "centered"
)

def read_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)
    
# streamlit page Title
st.title("🤖 ASK YOUR DATA - OLLAMA")

# Initialize chat history in streamlit session state
if "Chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
    
# initiate df in session state
if "df" not in st.session_state:
    st.session_state.df = None
    
    
    
# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["csv","xlsx","xls"])

if uploaded_file:
    st.session_state.df = read_data(uploaded_file)
    st.write("DataFrame Previews:")
    st.dataframe(st.session_state.df.head())
    

# Display Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])
        
# input field for user's message
user_prompt = st.chat_input("Ask LLM...")

if user_prompt:
    # add user's message to chat history and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user",
        "content": user_prompt})
    
    # loading LLM
    llm = ChatOllama(model="llama2",temperature=0.5)

    pandas_df_agent = create_pandas_dataframe_agent(llm, st.session_state.df, verbose = True, agent_type = AgentType.OPENAI_FUNCTIONS,
                                                allow_dangerous_code=True)

    message = [
        {"role":"system","content":"You are a helpful assistant"},
        *st.session_state.chat_history
    ]

    response = pandas_df_agent.invoke(user_prompt)

    assistant_response = response["output"]

    st.session_state.chat_history.append({"role":"assistant","content": assistant_response})

    #display LLM response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)