import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os

# Load environment variables
load_dotenv()

# Set up the Streamlit app
st.set_page_config(page_title="MathShift", page_icon="🔢", layout="wide")
st.title("MathShift - Decomposing problems into logical components")

# Get API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

# Initialize Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Useful for searching general information."
)

# Initialize math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solves math expressions step by step."
)

# Reasoning tool using a prompt template
prompt = """
You are an agent tasked with solving the user's mathematical question. 
Logically arrive at the solution and provide a detailed explanation, presented point-wise.

Question: {question}

Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_chain.run,
    description="Handles logic and reasoning questions."
)

# Initialize the agent with tools
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Set up chat session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your math assistant. How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input area for new question
question = st.text_area("Ask your math question here...")

if st.button("Find my Answer"):
    if question:
        with st.spinner("Generating Response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            
            # Streamlit callback for LangChain agent
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            try:
                # IMPORTANT: Pass the question string, not the whole message list
                response = assistant_agent.run(question, callbacks=[st_cb])
            except Exception as e:
                response = f"Error: {e}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.error("Please enter a question.")
