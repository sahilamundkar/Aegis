import streamlit as st

# Set page config at the very beginning
st.set_page_config(page_title="Aegis", page_icon="", layout="wide")

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
import os
from dotenv import load_dotenv
import backoff
import tiktoken

# Load environment variables
load_dotenv()

# Get GROQ API key from environment variable
#groq_api_key=os.environ['GROQ_API_KEY']

#if not os.environ['GROQ_API_KEY']:
#    os.environ['GROQ_API_KEY'] = input("Please enter your Groq API key: ")

# Function to verify API key
def verify_api_key(api_key):
    try:
        chat = ChatGroq(groq_api_key=api_key, model_name="Llama-3.1-70b-Versatile")
        response = chat.invoke("Test message")
        return True
    except Exception as e:
        return False

# Initialize session state for Groq API key
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = os.environ.get('GROQ_API_KEY', '')

# Prompt for Groq API key if not set
if not st.session_state.groq_api_key:
    api_key_input = st.empty()
    new_api_key = api_key_input.text_input("Please enter your Groq API key:", type="password")
    if new_api_key:
        if verify_api_key(new_api_key):
            st.session_state.groq_api_key = new_api_key
            os.environ['GROQ_API_KEY'] = new_api_key
            api_key_input.empty()
        else:
            st.error("Invalid API key. Please check your key and try again.")
            st.stop()

# Check if API key is set before proceeding
if not st.session_state.groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

st.image("aegis logo.JPG", use_column_width=150)

# Initialize LLM with retry decorator
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def create_llm():
    return ChatGroq(groq_api_key=st.session_state.groq_api_key, model_name="Llama-3.1-70b-Versatile")

llm = create_llm()

# Function to count tokens
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_prompt_with_history(history, questions_asked):
    conversation_history = "\n".join([f"User: {chat['question']}\nChatbot: {chat['answer']}" for chat in history])
        
    if questions_asked < 5:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are an AI assistant acting as an auditor to help a cybersecurity implementation engineer design the cybersecurity framework for their company using the ISO 27001 and ISO27002 standards.
            You have asked {questions_asked+1} questions so far.

            Conversation history:
            {conversation_history}

            Ask the next most appropriate question to understand the company better. Do not repeat any previous questions.
            The questions should be concise, restricted to one line and should cover only one topic.
            Format your questions as:
            Question {questions_asked+1}: [Your question here]

            Use the context and conversation history to inform your response. Provide the most accurate and relevant information possible.
            """),
            ("human", "Context: {context}\n\nUser Input: {input}")
        ])
    elif questions_asked == 5:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are an AI assistant acting as an auditor to help a cybersecurity implementation engineer design the cybersecurity framework for their company using the ISO 27001 and ISO27002 standards.
            You have asked {questions_asked+1} questions so far.

            Conversation history:
            {conversation_history}

            Based on the information provided, here are the key guidelines from ISO27001/ISO27002 for your company's cybersecurity framework:
            [Your comprehensive guidelines here, mention 10 most relevant guidelines] (While answering the guidelines, you should mention which parts/subsections/annex of which document(ISO27001 or ISO27002) you are referencing, be as descriptive as possible)
            Support your answer about each guideline by mentioning how your narrowed your search to that guideline using the information about the company (answers from the user). 

            Use the context and conversation history to inform your response. Provide the most accurate and relevant information possible.
            """),
            ("human", "Context: {context}\n\nUser Input: {input}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are an AI assistant acting as an auditor to answer the user's questions and write code if requested.

            Conversation history:
            {conversation_history}

            The user's question is the last item in the conversation.
            Please answer the user's query based on the information provided in the conversation history, the context, and your knowledge of ISO27001 and ISO27002 standards. Be specific and provide references to the relevant sections of the standards when appropriate.
            """),
            ("human", "Context: {context}\n\nUser Query: {input}")
        ])
    
    return prompt

@st.cache_resource
def load_or_create_embeddings():
    embeddings_file = "faiss_index"
    if os.path.exists(embeddings_file):
        embeddings = OllamaEmbeddings()
        vectors = FAISS.load_local(embeddings_file, embeddings, allow_dangerous_deserialization=True)
        # st.success("Loaded existing embeddings")
    else:
        embeddings = OllamaEmbeddings()
        loader = PyPDFDirectoryLoader("./ISO")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:20])
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local(embeddings_file)
        st.success("Created and saved new embeddings")
    return vectors

# Load embeddings at startup
vectors = load_or_create_embeddings()

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "full_conversation_history" not in st.session_state:
    st.session_state.full_conversation_history = []
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "Hello! I'm Aegis, an AI Cybersecurity Auditor and an expert on ISO27001 and ISO27002 documentation. I can help you to design a cybersecurity framework for your company. Please answer the questions that follow.\n\nQuestion 1: Can you describe your company's primary business activities and industries?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    st.session_state.conversation_history.append({"question": "", "answer": initial_message})
    st.session_state.full_conversation_history.append({"question": "", "answer": initial_message})
    st.session_state.questions_asked = 1
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "last_reset_time" not in st.session_state:
    st.session_state.last_reset_time = time.time()
if "company_info" not in st.session_state:
    st.session_state.company_info = None
if "show_api_limit_options" not in st.session_state:
    st.session_state.show_api_limit_options = False
if "last_unanswered_prompt" not in st.session_state:
    st.session_state.last_unanswered_prompt = None

def generate_response(prompt):
    conversation_history = "\n".join([f"User: {chat['question']}\nChatbot: {chat['answer']}" for chat in st.session_state.conversation_history])
    prompt_template = get_prompt_with_history(st.session_state.conversation_history, st.session_state.questions_asked)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    try:
        response = retrieval_chain.invoke({'input': prompt, 'context': conversation_history})
        reply = response['answer']
        
        st.session_state.conversation_history.append({"question": prompt, "answer": reply})
        st.session_state.full_conversation_history.append({"question": prompt, "answer": reply})
        if st.session_state.questions_asked <= 5:
            st.session_state.questions_asked += 1

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.token_count += num_tokens_from_string(reply)
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            st.session_state.last_unanswered_prompt = prompt
            handle_token_limit()
        else:
            st.error("An error occurred. Please try again.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to handle token limit
def handle_token_limit():
    st.session_state.show_api_limit_options = True
    st.rerun()

def restart_conversation():
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.session_state.full_conversation_history = []
    st.session_state.questions_asked = 0
    st.session_state.token_count = 0
    st.session_state.last_reset_time = time.time()
    st.session_state.company_info = None
    st.session_state.show_api_limit_options = False
    st.session_state.last_unanswered_prompt = None
    initial_message = "Hello! I'm Aegis, an AI Cybersecurity Auditor and an expert on ISO27001 and ISO27002 documentation. I can help you to design a cybersecurity framework for your company. Please answer the questions that follow.\n\nQuestion 1: Can you describe your company's primary business activities and industries?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    st.session_state.conversation_history.append({"question": "", "answer": initial_message})
    st.session_state.full_conversation_history.append({"question": "", "answer": initial_message})
    st.session_state.questions_asked = 1
    st.rerun()

def remember_company_inquiry():
    st.warning("Conversation history has been trimmed to stay within API limits. Company information and guidelines have been retained.")
    # Keep only the first 5 questions and answers, and the guidelines
    st.session_state.company_info = st.session_state.conversation_history[:6]
    st.session_state.conversation_history = st.session_state.company_info
    st.session_state.messages = [{"role": "assistant" if i % 2 == 0 else "user", "content": msg["answer"] if i % 2 == 0 else msg["question"]} 
                                 for i, msg in enumerate(st.session_state.full_conversation_history)]
    st.session_state.token_count = sum(num_tokens_from_string(msg["content"]) for msg in st.session_state.messages)
    st.session_state.last_reset_time = time.time()
    st.session_state.show_api_limit_options = False
    
    # Answer the last unanswered prompt
    if st.session_state.last_unanswered_prompt:
        generate_response(st.session_state.last_unanswered_prompt)
        st.session_state.last_unanswered_prompt = None
    
    st.rerun()

# Display API limit options if needed
if st.session_state.show_api_limit_options:
    st.warning("API rate limit reached. Please choose an option:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restart the conversation"):
            restart_conversation()
    with col2:
        if st.button("Retain the company information"):
            remember_company_inquiry()

# React to user input
if prompt := st.chat_input("Your response here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.full_conversation_history.append({"question": prompt, "answer": ""})
    
    # Check if a minute has passed since the last reset
    current_time = time.time()
    if current_time - st.session_state.last_reset_time >= 60:
        st.session_state.token_count = 0
        st.session_state.last_reset_time = current_time

    st.session_state.token_count += num_tokens_from_string(prompt)

    # Check if token count is approaching the limit (e.g., 5500 tokens to leave some buffer)
    if st.session_state.token_count > 5500:
        st.session_state.last_unanswered_prompt = prompt
        handle_token_limit()
    else:
        with st.spinner("Generating response..."):
            generate_response(prompt)

# Option to clear the conversation history
if st.sidebar.button("Clear Conversation History"):
    restart_conversation()