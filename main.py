import os
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv,find_dotenv
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_verbose, set_debug
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
set_debug(True)
set_verbose(True)


load_dotenv(find_dotenv())

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = "mekari-test"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    convert_system_message_to_human=True
)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

custom_prompt_template = """
Thees are Document of Google Store reviews for a music streaming application (Spotify) sourced from various users. The management is currently
facing difficulties in extracting actionable insights from these reviews, Please answer this Question.

Document: {document}
Question: {question}
Answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["document", "question"],
    template=custom_prompt_template
)

llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain,document_variable_name="document"
)

qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_documents_chain
)


# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

def get_tfidf_matrix(texts):
    return vectorizer.fit_transform(texts)

def evaluate_response(response, reference_response):
    # Create a TF-IDF matrix for both responses
    tfidf_matrix = get_tfidf_matrix([response, reference_response])
    # Compute the cosine similarity between the two responses
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).item()
    return similarity

# Streamlit app
st.title("Chat with Bot")

# Initialize session state for storing chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to generate the response
def generate_response(query):
    response = qa_chain.invoke(query).get("result")
    return response

# Display the chat history
for message in st.session_state.messages:
    if message["is_user"]:
        st.text_input("User:", value=message["content"], key=f"user_{message['index']}")
    else:
        st.text_area("Bot:", value=message["content"], key=f"bot_{message['index']}")
        # Display similarity score if reference response is available
        if 'reference_response' in st.session_state:
            similarity = evaluate_response(message["content"], st.session_state.reference_response)
            st.write(f"Similarity Score: {similarity:.4f}")

# User query input
query = st.text_input("Enter your query:", key="user_input")

if st.button("Send"):
    if query:
        # Store user message
        st.session_state.messages.append({"content": query, "is_user": True, "index": len(st.session_state.messages)})
        # Generate response
        response = generate_response(query)
        # Store bot response
        st.session_state.messages.append({"content": response, "is_user": False, "index": len(st.session_state.messages)})
        # Update the input field (this works around the disabled issue)
        st.experimental_rerun()
    else:
        st.write("Please enter a query.")

# Reference response input
reference_response = st.text_input("Enter reference response for evaluation:", key="reference_response_input")

if reference_response:
    st.session_state.reference_response = reference_response
else:
    # Clear reference response if the input is empty
    if 'reference_response' in st.session_state:
        del st.session_state.reference_response