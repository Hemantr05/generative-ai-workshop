from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


file_path = "data/Harry Potter and the Sorcerers Stone.pdf"


loader = PyPDFLoader(file_path)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


docs = [
    Document(
        page_content=page.page_content,
        metadata=page.metadata
    )
    for page in loader.lazy_load()
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(splits, embedding_function=embeddings, persist_directory="./chroma_db")

if __name__ == "__main__":
    query = "What is life?"
    docs = vector_store.similarity_search_with_score(query)

    for d in docs:
        print("="*50)
        print(d)