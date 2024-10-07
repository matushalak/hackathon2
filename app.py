import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from custom_tools import get_pokemon_types, search_duckduckgo
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ChatMessageHistory
import numpy as np

# Ensure that st.set_page_config() is the first Streamlit command
st.set_page_config(
    page_title="Open AI Agent",
    page_icon=":sparkles:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Ensure active_prompt is initialized
if "active_prompt" not in st.session_state:
    st.session_state.active_prompt = "companion"

# Ensure chat_history is initialized
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# Function to toggle the active prompt, change the title, and clear the chat history
def toggle_prompt():
    if st.session_state.active_prompt == "companion":
        st.session_state.active_prompt = "journaling"
    else:
        st.session_state.active_prompt = "companion"

    # Clear chat history when toggling
    st.session_state.chat_history = []

# Function to read valence and arousal from a txt file
def read_valence_arousal(file_path=r"C:\Users\andre\OneDrive\Documents\Personal\Hackathon\val_ar.txt"):
    try:
        # with open(file_path, 'r') as f:
        #     data = f.readlines()
        #     valence = int(data[0].split(":")[1].strip())
        #     arousal = int(data[1].split(":")[1].strip())
        val, ar = np.loadtxt(file_path, skiprows=1)
        valence = round(val.astype(float), 2) 
        arousal = round(ar.astype(float), 2)

        return valence, arousal
    except Exception as e:
        st.error(f"Error reading valence and arousal from file: {e}")
        return None, None

# Define the prompts for the two chatbot modes (scores are used but not mentioned)
def get_prompt_by_name(name):
    if name == "companion":
        return ChatPromptTemplate.from_messages([
            ("system", '''
            You are a chatbot designed to provide companionship and emotional support for users who may be feeling lonely. You will receive emotional signals (valence and arousal), but you should never mention these scores to the user. Instead, use them to guide the conversation in a subtle way. Your tone should be warm, empathetic, and supportive.

            - When the user is feeling stressed or overwhelmed (high arousal, low valence), focus on calming and reassuring conversation topics.
            - When the user is feeling down (low arousal, low valence), offer gentle support, asking them how they’re feeling and offering to listen without judgment.
            - When the user is feeling happy (high valence), engage them in positive conversation, asking about things that bring them joy.
            - When the user’s mood is neutral or balanced, keep the conversation light and supportive, focusing on connection.

            Never mention the emotional scores, but use them to adjust your tone and suggestions. Begin by acknowledging the user's presence and checking in on how they’re feeling.
            '''),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    elif name == "journaling":
        return ChatPromptTemplate.from_messages([
            ("system", '''
            You are a journaling assistant designed to help users explore and express their emotions. You will receive emotional signals (valence and arousal), but you should never mention these scores to the user. Instead, subtly use them to guide journaling prompts based on how the user might be feeling.

            - When the user is feeling stressed or overwhelmed (high arousal, low valence), offer reflective prompts to help them process intense emotions.
            - When the user is feeling low (low arousal, low valence), offer soothing, supportive prompts, encouraging them to gently explore what’s on their mind.
            - When the user is feeling happy (high valence), encourage positive reflections, creative writing, or aspirational thoughts.
            - When the user’s emotions are balanced or neutral, suggest prompts that explore their current mood or general reflections.

            Never mention the emotional scores, but use them to guide your tone and prompts. Begin by asking the user how they are feeling and suggest a journaling topic aligned with their emotional state.
            '''),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

# Dynamically change the title based on the active prompt
title = "Companion Chatbot" if st.session_state.active_prompt == "companion" else "Journaling Assistant"
st.title(title)

# Center the toggle button at the top
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button(f"Switch to {'Journaling Assistant' if st.session_state.active_prompt == 'companion' else 'Companion Chatbot'}"):
    toggle_prompt()
    st.experimental_rerun()  # Rerun the script to update the title and prompt
st.markdown("</div>", unsafe_allow_html=True)

# Get the active prompt
prompt = get_prompt_by_name(st.session_state.active_prompt)

# Add memory to the chat for conversation history
formatted_chat_history = ChatMessageHistory()
for message in st.session_state.chat_history:
    if message["role"] == "user":
        formatted_chat_history.add_user_message(message["content"])
    elif message["role"] == "assistant":
        formatted_chat_history.add_ai_message(message["content"])
memory = ConversationBufferMemory(
    chat_memory=formatted_chat_history,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

# Load model from Open AI
openai_key = st.secrets["OPEN_AI_KEY"]
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    openai_api_key=openai_key,
    temperature=0.2,
    streaming=True,
)
model = llm.bind(functions=[format_tool_to_openai_function(f) for f in [get_pokemon_types, search_duckduckgo]])

# Create chain
chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    )
    | prompt
    | model
    | OpenAIFunctionsAgentOutputParser()
)

# Create executor
executor = AgentExecutor(
    agent=chain,
    tools=[get_pokemon_types, search_duckduckgo],
    verbose=False,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    memory=memory,
)

# Function to display chat
def display_chat():
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

# Function to get model answer
def create_answer(question):
    # Read valence and arousal from file
    valence, arousal = read_valence_arousal()

    if valence is not None and arousal is not None:
        combined_input = f"The user said: {question}. Valence: {valence}, Arousal: {arousal}."
    else:
        combined_input = f"The user said: {question}."

    result = executor.invoke({"input": combined_input})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["output"] + f' [{valence}, {arousal}]',
        }
    )

# Handle user input
if question := st.chat_input(placeholder="Let's chat"):
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )
    create_answer(question)
    display_chat()