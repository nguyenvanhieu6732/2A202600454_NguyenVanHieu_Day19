import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Pydantic schemas for structured extraction
class Triple(BaseModel):
    subject: str = Field(description="Thực thể chính (Node 1), ví dụ: 'OpenAI'")
    predicate: str = Field(description="Mối quan hệ (Edge), luôn dùng chữ HOA và dấu _, ví dụ: 'FOUNDED_BY', 'LOCATED_IN'")
    object: str = Field(description="Thực thể liên quan (Node 2), ví dụ: 'Sam Altman'")

class TriplesExtraction(BaseModel):
    triples: List[Triple] = Field(description="Danh sách các bộ ba (triples) trích xuất được từ văn bản")

class Neo4jGraphManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_triples(self, triples):
        with self.driver.session() as session:
            for t in triples:
                # Ensure predicate is safe (only A-Z and _)
                predicate = "".join(c for c in t.predicate.upper() if c.isalnum() or c == '_').replace(' ', '_')
                if not predicate:
                    continue
                query = f"""
                MERGE (n1:Entity {{name: $subject}})
                ON CREATE SET n1.id = $subject
                MERGE (n2:Entity {{name: $object}})
                ON CREATE SET n2.id = $object
                MERGE (n1)-[:{predicate}]->(n2)
                """
                session.run(query, subject=t.subject, object=t.object)
                
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
    def add_node_embeddings(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("Fetching all nodes to generate embeddings...")
        with self.driver.session() as session:
            # Create vector index
            try:
                session.run('''
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS 
                FOR (n:Entity) ON (n.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
                ''')
            except Exception as e:
                print("Vector index exist or error:", e)
                
            # Fetch nodes without embeddings
            result = session.run("MATCH (n:Entity) WHERE n.embedding IS NULL RETURN n.name AS name")
            node_ids = [record["name"] for record in result]
            
            if not node_ids:
                print("No new nodes to embed.")
                return
                
            print(f"Generating embeddings for {len(node_ids)} nodes...")
            # Batch API
            batch_size = 500
            for i in range(0, len(node_ids), batch_size):
                batch_ids = node_ids[i:i+batch_size]
                vectors = embeddings.embed_documents(batch_ids)
                
                # Update Neo4j
                query = '''
                UNWIND $data AS row
                MATCH (n:Entity {name: row.name})
                SET n.embedding = row.embedding
                '''
                data = [{"name": _id, "embedding": vec} for _id, vec in zip(batch_ids, vectors)]
                session.run(query, data=data)
            print("Embeddings generation complete!")

def main():
    print("Connecting to Neo4j...")
    graph = Neo4jGraphManager(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    print("Clearing old data in Neo4j...")
    graph.clear_database()

    # Read corpus
    corpus_file = "data/corpus.txt"
    if not os.path.exists(corpus_file):
        print(f"File {corpus_file} không tồn tại!")
        return

    with open(corpus_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    print(f"Total chunks to process: {len(chunks)}")

    # LLM init
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(TriplesExtraction)

    total_triples = 0
    print(f"Processing all {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        try:
            prompt = f"Trích xuất tất cả các thực thể (entities) và mối quan hệ (relationships) từ đoạn văn sau. Trả về dưới dạng triples. Đoạn văn: \n{chunk}"
            result = structured_llm.invoke(prompt)
            if result and result.triples:
                graph.insert_triples(result.triples)
                total_triples += len(result.triples)
        except Exception as e:
            print(f"Error on chunk {i+1}: {e}")
            
    print("Adding Embeddings for nodes in Neo4j...")
    graph.add_node_embeddings()

    graph.close()
    print(f"Done! Inserted {total_triples} relationships into Neo4j.")

if __name__ == "__main__":
    main()
