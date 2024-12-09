from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class QueryVectorDB():
    def __init__(self,
                 file_path: str):
        
        self.file_path = file_path


        # create the open-source embedding function
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        self.loaded_document = None
        self.splits = None

        self.vector_store = None

    def load_pdf(self):
        # Load the PDF
        loader = PyPDFLoader(self.file_path)

        # Load all pages
        self.loaded_document = loader.load()

    def create_split(self, chunk_size: int=1000, chunk_overlap: int=200):
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.splits = text_splitter.split_documents(self.loaded_document)

    def load_vectordb(self, persist_directory: str):
        # Initialize Chroma and persist data
        self.vector_store = Chroma.from_documents(self.splits, self.embedding_function, persist_directory=persist_directory)
        self.vector_store.persist()

    def create_vectordb(self, persist_directory: str='./chroma_db'):
        self.load_pdf()
        self.create_split()
        self.load_vectordb(persist_directory)

    def query(self, q: str):
        responses = self.vector_store.similarity_search_with_score(q)
        return self.postprocess(responses)
    
    def postprocess(self, responses):
        final = {}
        for idx, response in enumerate(responses):
            content, score = response
            page_content = content.page_content
            metadata = content.metadata
            final[idx] = {
                "page_content": page_content,
                "metadata": metadata.update({"score": score})
            }
        return final