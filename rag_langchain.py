import chromadb
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader

# Load and combine both PDFs using PyPDFLoader
menu_loader = PyPDFLoader("data/Menu.pdf")
allergy_loader = PyPDFLoader("data/Liste_allergenes.pdf")

menu_docs = menu_loader.load()
allergy_docs = allergy_loader.load()

all_docs = menu_docs + allergy_docs


class OllamaEmbeddingWrapper:
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(input)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

# --- Config ---
COLLECTION_NAME = "cours_rag_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2:latest"

# Simple chunker: splits texts into chunks (you can replace with a LangChain splitter if preferred)
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc in docs:
        text = doc.page_content
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i : i + chunk_size])
    print(f"Created {len(chunks)} chunks from all documents")
    return chunks

def initialize_vectorstore_from_chunks(client, collection_name, embedding_function, chunks):
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )
    
    if collection.count() == 0:
        print("Adding chunks to collection...")
        collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
    else:
        print("Collection already populated, skipping add.")
    
    return collection

if __name__ == "__main__":
    print("Initialisation des composants LangChain...")

    # Init embeddings and client
    ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    wrapped_embeddings = OllamaEmbeddingWrapper(ollama_embeddings)

    client = chromadb.PersistentClient(path="./chroma_db")

    # Chunk all docs together
    chunks = chunk_documents(all_docs)

    # Initialize or update collection with combined chunks
    collection = initialize_vectorstore_from_chunks(client, COLLECTION_NAME, wrapped_embeddings, chunks)

    # Setup vectorstore with wrapped embeddings (important!)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=wrapped_embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=LLM_MODEL)

    template = """Réponds à la question en te basant uniquement sur le contexte suivant:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- Chatbot RAG avec LangChain ---")
    print("Posez des questions sur le document. Tapez 'exit' pour quitter.")

    while True:
        user_question = input("\nVous: ")
        if user_question.lower() == "exit":
            break

        try:
            docs = retriever.get_relevant_documents(user_question)
            print("\n[DEBUG] Contexte récupéré :")
            for i, doc in enumerate(docs):
                print(f"--- Chunk {i+1} ---")
                print(doc.page_content[:300], "\n")

            answer = rag_chain.invoke(user_question)
            print(f"\rAssistant: {answer}")

        except Exception as e:
            print(f"Une erreur est survenue : {e}")
