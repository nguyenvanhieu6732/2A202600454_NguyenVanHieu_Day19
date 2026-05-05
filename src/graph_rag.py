import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class EntityExtraction(BaseModel):
    entity: str = Field(description="Thực thể chính (công ty, tổ chức) được hỏi trong câu hỏi")

class GraphRAGQuery:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.entity_llm = self.llm.with_structured_output(EntityExtraction)

    def extract_entity(self, query: str) -> str:
        prompt = f"Trích xuất tên thực thể chính (một từ hoặc cụm từ) được đề cập trong câu hỏi sau:\n{query}"
        res = self.entity_llm.invoke(prompt)
        return res.entity if res else ""

    def get_seed_nodes(self, entity: str, top_k: int = 2) -> list[str]:
        # Generate embedding for the entity string
        query_vector = self.embeddings.embed_query(entity)
        
        # Use Vector Search to find closest nodes
        query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_vector)
        YIELD node, score
        RETURN node.id AS id, score
        """
        seed_nodes = []
        with self.driver.session() as session:
            try:
                result = session.run(query, top_k=top_k, query_vector=query_vector)
                seed_nodes = [record["id"] for record in result if record["score"] > 0.5]
            except Exception as e:
                print("Vector search error (falling back to text search):", e)
                
        # Fallback to text match if vector search fails or returns nothing
        if not seed_nodes:
            query_fallback = """
            MATCH (n:Entity)
            WHERE n.id CONTAINS $entity OR toLower(n.id) CONTAINS toLower($entity)
            RETURN n.id AS id LIMIT $top_k
            """
            with self.driver.session() as session:
                result = session.run(query_fallback, entity=entity, top_k=top_k)
                seed_nodes = [record["id"] for record in result]
                
        return seed_nodes

    def get_graph_context(self, seed_nodes: list[str]) -> str:
        if not seed_nodes:
            return "Không tìm thấy thông tin đồ thị."
            
        # Lấy thông tin trong phạm vi 2-hops từ các seed nodes
        query = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.id IN $seed_nodes OR m.id IN $seed_nodes
        RETURN n.id AS source, type(r) AS relation, m.id AS target
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query, seed_nodes=seed_nodes)
            triples = [f"({record['source']}, {record['relation']}, {record['target']})" for record in result]
        
        if not triples:
            return "Không tìm thấy thông tin đồ thị từ các đỉnh bắt đầu."
        return "\n".join(triples)

    def answer(self, query: str) -> str:
        # Bước 1: Trích xuất thực thể
        entity = self.extract_entity(query)
        print(f"[GraphRAG] Extracted Entity: {entity}")
        
        if not entity:
            return "Không thể xác định thực thể trong câu hỏi."

        # Bước 2: Tìm Seed Nodes và Truy vấn Neo4j
        seed_nodes = self.get_seed_nodes(entity)
        print(f"[GraphRAG] Seed Nodes: {seed_nodes}")
        
        graph_context = self.get_graph_context(seed_nodes)
        
        # Bước 3: Tổng hợp (Textualization) và Trả lời
        prompt = f"""
Bạn là một trợ lý AI trả lời câu hỏi dựa vào sơ đồ tri thức.
Thông tin đồ thị tri thức trích xuất được (các bộ ba chủ ngữ - vị ngữ - tân ngữ):
{graph_context}

Dựa vào dữ liệu đồ thị trên, hãy trả lời câu hỏi:
Câu hỏi: {query}
"""
        response = self.llm.invoke(prompt)
        return response.content

if __name__ == "__main__":
    rag = GraphRAGQuery()
    question = "OpenAI được thành lập bởi ai?"
    print(f"Q: {question}")
    print(f"A: {rag.answer(question)}")
