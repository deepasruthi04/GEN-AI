import os
import numpy as np
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq



pdf_path = r"C:\Users\deepasruthi.ramesh\Documents\ALL gen ai\RAG with textualData\Internet of Things_ Architectures, Protocols and Standards ( PDFDrive ).pdf"
loader = PyPDFLoader(pdf_path)
pdf_pages = loader.load()

chunk_size = 1000
chunk_overlap = 50

splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunked_docs = splitter.split_documents(pdf_pages)

unique_docs = []
seen_texts = set()
for doc in chunked_docs:
    if doc.page_content not in seen_texts:
        unique_docs.append(doc)
        seen_texts.add(doc.page_content)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_directory = "./db/chroma"

if not os.path.exists(persist_directory):
    vectordb = Chroma.from_documents(
        documents=unique_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
else:
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

os.environ["GROQ_API_KEY"] =  "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"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=250)

try:
    test_response = llm.invoke("Hello")
    print("Groq LLM connection successful.")
except Exception as e:
    print("Connection test failed:", e) 

def ask_iot_question(question):
    system_prompt = (
        "You are an assistant for question-answering only for the context based on IoT data. "
        "Use the following data given to you to answer the question. "
        "If the question doesn't relate to IoT or the given context, say 'I don't know'. "
        "Keep the answer concise up to 4 sentences."
    )

    docs = vectordb.similarity_search(question, k=10)
    context = "\n".join([doc.page_content for doc in docs])

    system_message = SystemMessage(content=system_prompt + "\n\nContext:\n" + context)
    user_message = HumanMessage(content=question)

    try:
        response_obj = llm.invoke([system_message, user_message])
        if hasattr(response_obj, "content"):
            return response_obj.content
        elif isinstance(response_obj, dict) and "content" in response_obj:
            return response_obj["content"]
        else:
            return str(response_obj)
    except Exception as e:
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    while True:
        q = input("Enter IoT question (or 'exit'): ")
        if q.lower() == "exit":
            break
        ans = ask_iot_question(q)
        print(f"Answer:\n{ans}\n")
