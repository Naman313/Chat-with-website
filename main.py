import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


os.environ["OPENAI_API_KEY"] = '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'



def get_vectorstore_from_url(url):

    #get the text from html
    loader= WebBaseLoader(url)
    documents= loader.load()
    text_splitter= RecursiveCharacterTextSplitter()
    document_chunks= text_splitter.split_documents(documents)
    # return document_chunks
    #crete a vector store

    vector_store= Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store



def get_context_retriever_chain(vector_store):
    llm= ChatOpenAI()
    retriever= vector_store.as_retriever()

    prompt= ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain= create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm= ChatOpenAI()
    prompt= ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name= "chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain= create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_Input):
     #create conversation chain 
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    #response= get_response(query)
    response= conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": query
    })

    return response['answer']




#App config
st.set_page_config(page_title="Chat with websites ", page_icon="")
st.title("Chat with websites ")

#Sidebar
with st.sidebar:
    st.header("Settings")
    website= st.text_input("Website URL")

if website is None or website== "":
    st.info("Please enter a valid URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= [
            # AIMessage(content="Hello world"),
            ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store= get_vectorstore_from_url(website)
    # vector_store= get_vectorstore_from_url(website)
    # with st.sidebar:
    #     st.write(vector_store)

    #create conversation chain 
    # retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    # conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

      #User Input
    query= st.chat_input("Type your message here.... ")
    if query is not None and query!= "":
        response= get_response(query)
        # st.write(response)
        st.session_state.chat_history.append(AIMessage(content=query))
        st.session_state.chat_history.append(HumanMessage(content= response))
        
        # retrieved_documents= retriever_chain.invoke({
        #     "chat-history": st.session_state.chat_history,
        #     "input": query
        # })
        # st.write(retrieved_documents)


        #Conversion
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)






