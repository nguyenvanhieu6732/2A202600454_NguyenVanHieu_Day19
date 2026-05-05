import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class FlatRAGQuery:
    def __init__(self, corpus_path="data/corpus.txt", db_dir="./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Load or initialize vector store
        if os.path.exists(db_dir):
            self.vectorstore = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
        else:
            self.vectorstore = self._build_index(corpus_path, db_dir)

    def _build_index(self, corpus_path, db_dir):
        print("[FlatRAG] Building Vector Index...")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"File {corpus_path} không tồn tại!")
            
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        docs = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = Chroma.from_documents(docs, self.embeddings, persist_directory=db_dir)
        print(f"[FlatRAG] Built index with {len(docs)} chunks.")
        return vectorstore

    def answer(self, query: str) -> str:
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""
Bạn là một trợ lý AI. Dựa vào thông tin văn bản dưới đây, hãy trả lời câu hỏi:
Văn bản: {context}

Câu hỏi: {query}
"""
        res = self.llm.invoke(prompt)
        return res.content

if __name__ == "__main__":
    rag = FlatRAGQuery()
    question = "OpenAI được thành lập bởi ai?"
    print(f"Q: {question}")
    print(f"A: {rag.answer(question)}")
