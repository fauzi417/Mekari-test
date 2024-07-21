# Chatbot Application with Streamlit and Pinecone

## Overview

This project is a chatbot application built with Streamlit, LangChain, Pinecone, and Google Generative AI. The application allows users to interact with a chatbot that answers questions based on a set of reviews. It also includes functionality to evaluate the quality of responses using cosine similarity.

## Features

- **Interaction with Chatbot:** Users can input queries and receive responses from a chatbot.
- **Quality Evaluation:** Users can input reference responses to evaluate the quality of the chatbot’s answers using cosine similarity.
- **Data Storage:** The chatbot uses Pinecone to store and retrieve vectorized document embeddings.

## Requirements

To run this application, you'll need to install the following packages:

- `langchain`
- `pinecone-client`
- `python-dotenv`
- `streamlit`
- `scikit-learn`
- `langchain_pinecone`
- `langchain_google_genai`

## Setup

1. **Create Pinecone Index:**

   The Pinecone index is created if it does not already exist. This index is used to store vector embeddings of documents.

   ```python
   index_name = "mekari-test"
   if index_name not in pc.list_indexes().names():
       pc.create_index(
           name=index_name,
           dimension=768,
           metric="euclidean",
           spec=ServerlessSpec(cloud='aws', region='us-east-1')
       )
   ```

2. **Initialize Embeddings and Language Model:**

   The embeddings are created using `HuggingFaceEmbeddings`, and a language model (`ChatGoogleGenerativeAI`) is initialized to generate responses.

   ```python
   embeddings = HuggingFaceEmbeddings()
   llm = ChatGoogleGenerativeAI(
       model="gemini-1.5-pro",
       temperature=0,
       google_api_key=os.environ.get("GOOGLE_API_KEY")
   )
   ```

3. **Configure Retrieval and QA Chains:**

   The vector store is configured to use Pinecone, and a retrieval-based QA chain is set up with a custom prompt template.

   ```python
   vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
   retriever = vectorstore.as_retriever()
   custom_prompt = PromptTemplate(
       input_variables=["document", "question"],
       template=custom_prompt_template
   )
   llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
   combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="document")
   qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_documents_chain)
   ```

4. **Cosine Similarity Evaluation:**

   Responses can be evaluated against a reference response using cosine similarity calculated on TF-IDF vectors.

   ```python
   def evaluate_response(response, reference_response):
       tfidf_matrix = get_tfidf_matrix([response, reference_response])
       similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).item()
       return similarity
   ```

5. **Streamlit UI:**

   The Streamlit app provides an interface for interacting with the chatbot, viewing responses, and entering reference responses for evaluation.

   ```python
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
           if 'reference_response' in st.session_state:
               similarity = evaluate_response(message["content"], st.session_state.reference_response)
               st.write(f"Similarity Score: {similarity:.4f}")
   
   # User query input
   query = st.text_input("Enter your query:", key="user_input")
   if st.button("Send"):
       if query:
           st.session_state.messages.append({"content": query, "is_user": True, "index": len(st.session_state.messages)})
           response = generate_response(query)
           st.session_state.messages.append({"content": response, "is_user": False, "index": len(st.session_state.messages)})
           st.experimental_rerun()
       else:
           st.write("Please enter a query.")
   
   # Reference response input
   reference_response = st.text_input("Enter reference response for evaluation:", key="reference_response_input")
   if reference_response:
       st.session_state.reference_response = reference_response
   else:
       if 'reference_response' in st.session_state:
           del st.session_state.reference_response
   ```

## Running the Application

To start the application, ensure all dependencies are installed and run the Streamlit app:

```bash
streamlit run main.py
```

---
