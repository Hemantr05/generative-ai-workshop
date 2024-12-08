from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

file_path = "data/Harry Potter and the Sorcerers Stone.pdf"

# Load the PDF
loader = PyPDFLoader(file_path)

# Load all pages
docs = loader.load()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize Chroma and persist data
vector_store = Chroma.from_documents(
    splits,
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
vector_store.persist()

# Query the vector store
if __name__ == "__main__":
    query = "What is life?"
    docs = vector_store.similarity_search_with_score(query)

    for d in docs:
        print("=" * 50)
        print(d)
