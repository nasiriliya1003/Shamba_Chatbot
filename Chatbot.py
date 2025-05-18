import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Optional: suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load your CSV file
csv_file = 'shamba_chat.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

shamba_data = pd.read_csv(csv_file)

# Combine rows into plain text for context
context_data = []
for _, row in shamba_data.iterrows():
    context = " ".join([f"{col}: {row[col]}" for col in shamba_data.columns])
    context_data.append(context)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Initialize Chroma vector store (persistent so it's reusable)
vectorstore = Chroma(
    collection_name="shamba_data_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_store"
)

# Add documents to vectorstore
vectorstore.add_texts(context_data)

# Create retriever from the vectorstore
retriever = vectorstore.as_retriever()

# Prompt Template
template = """You are an agricultural expert. Use the context to answer the question.
If the answer is not in the context, say you don't know. Be clear and concise.

Context: {context}

Question: {question}

Answer:"""
rag_prompt = PromptTemplate.from_template(template)

# Load Ollama LLM (ensure Ollama and a supported model like `qwen2.5:3b` or `mistral` is running)
llm = ChatOllama(model="qwen2.5:3b")  # Change model if needed

# Build RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# CLI Interface
print("🌾 Shamba Chatbot (Type 'exit' to quit)")

while True:
    print ('#########'*15)
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    if not user_input:
        print("⚠️ Please enter a valid question.")
        continue

    try:
        response = rag_chain.invoke(user_input)
        print('-----' * 20)
        print("Shamba Chatbot:", response)
    except Exception as e:
        print("❌ Error:", str(e))
