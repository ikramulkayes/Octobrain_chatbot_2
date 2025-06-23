import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re

# Load environment variables from .env file
load_dotenv()

# Access the API key from .env file
api_key = os.getenv("API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_api_url = os.getenv("QDRANT_API_URL")

# Configure the generative AI client
genai.configure(api_key=api_key)

# Initialize the LangChain chat model (using Gemini)
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Initialize ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)

# Initialize Qdrant client for the general-purpose "octobrain" collection
qdrant_client = QdrantClient(
    url=qdrant_api_url, 
    api_key=qdrant_api_key
)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to clean HTML tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Prompt template for answering questions about Octobrain
octobrain_prompt = PromptTemplate(
    input_variables=["user_query", "search_results"],
    template="""
You are an expert assistant specializing in the Octobrain platform. Based on the user's query and the relevant information retrieved from the database, provide a detailed and accurate response.

User Query:
{user_query}

Relevant Information:
{search_results}

Answer the query in a clear and concise manner, ensuring the response is helpful and directly addresses the user's question.
"""
)

# LLM chain for answering questions about Octobrain
octobrain_chain = LLMChain(
    llm=chat_model,
    prompt=octobrain_prompt
)

def search_octobrain(query_text, history=None, limit=5):
    """
    Given a query and optional history, generate an embedding and search the "octobrain" collection in Qdrant.
    """
    try:
        # Combine history with the current query if history is provided
        if history:
            combined_query = f"{history}\n\n{query_text}"
        else:
            combined_query = query_text
        
        # Generate embedding for the combined query
        query_embedding = model.encode(combined_query)
        
        # Perform the search in Qdrant
        search_results = qdrant_client.search(
            collection_name="general_data",
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        if search_results:
            return [result.payload for result in search_results if result.score >= 0.3]
        return []
    except Exception as e:
        st.error(f"Error searching Octobrain database: {e}")
        return []

# Function to extract relevant history
def get_relevant_history(messages, max_turns=3):
    """
    Extract the last few turns of conversation history to provide context for the search.
    """
    history = []
    for message in reversed(messages):
        if message["role"] == "user" or message["role"] == "assistant":
            history.append(message["content"])
        if len(history) >= max_turns * 2:  # Each turn includes a user and assistant message
            break
    return "\n".join(reversed(history))  # Reverse to maintain chronological order

# Streamlit app layout
st.title("Octobrain Knowledge Chatbot")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # System message for context
    system_msg = {"role": "system", "content": "You are an expert assistant specializing in the Octobrain platform."}
    st.session_state.messages.append(system_msg)
    
    initial_greeting = ("Hello! I'm your Octobrain assistant. Ask me anything about the Octobrain platform, "
                        "its features, clubs, workshops, or any other topic, and I'll provide you with detailed answers!")
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about Octobrain..."):
    # Append the user prompt to session state as a dict
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Initialize a variable for response_text
    response_text = ""
    
    # Extract relevant history
    relevant_history = get_relevant_history(st.session_state.messages)
    
    # Search the Octobrain database with history
    search_results = search_octobrain(prompt, history=relevant_history)
    
    if search_results:
        # Format search results for the prompt
        results_text = "\n\n".join(
            f"Content: {clean_html(result.get('text_data', ''))}" for result in search_results
        )
        
        # Generate a response using the LLM chain
        response_data = octobrain_chain.invoke({
            "user_query": prompt,
            "search_results": results_text
        })
        response_text = response_data.get('text', "Sorry, I couldn't generate a response.")
    else:
        response_text = "I couldn't find any relevant information in the database. Could you rephrase your question or provide more details?"
    
    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Append the assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Update conversation memory
    memory.chat_memory.add_user_message(prompt)
    memory.chat_memory.add_ai_message(response_text)

# Create a sample output file
with open('octobrain_knowledge_chatbot.py', 'w') as f:
    f.write("""
# Octobrain Knowledge Chatbot
# This file contains the code for a Streamlit-based chatbot that answers questions about the Octobrain platform
# Uses Qdrant Cloud for vector search
# Run with: streamlit run octobrain_knowledge_chatbot.py --server.enableCORS=false
""")