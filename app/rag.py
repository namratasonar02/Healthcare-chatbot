import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from app.config import settings
from app.embeddings import get_embeddings
from app.llm import get_llm
from app.utils import logger

class RAGService:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.chroma_path = settings.chroma_path
        self.vector_store = self._init_vector_store()
        self.llm = get_llm()
        self.prompt = self._get_prompt_template()

    def _init_vector_store(self) -> Chroma:
        return Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)

    def _get_prompt_template(self) -> PromptTemplate:
        template = """You are a healthcare document assistant.

Answer ONLY using the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not guess.
3. Do not hallucinate.
4. If the answer is not present in the context, respond exactly:

'I could not find this information in the provided documents.'

5. Do not provide medical diagnosis.
6. Do not provide treatment recommendations.
7. Keep answers concise and professional.
8. Always reference supporting document sources.

Context:
{context}

Question:
{question}

Answer:"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def ingest_documents(self) -> dict:
        logger.info("Starting ingestion process...")
        
        # Clear existing vector store collection to avoid duplicates and handle model/embedding changes
        try:
            self.vector_store.delete_collection()
            logger.info("Cleared existing vector store collection.")
        except Exception as e:
            logger.warning(f"Could not clear vector store collection: {e}")
        
        # Re-initialize vector store
        self.vector_store = self._init_vector_store()
        
        data_path = settings.data_path
        file_pattern = os.path.join(data_path, "*.txt")
        files = glob.glob(file_pattern)
        
        if not files:
            logger.warning(f"No .txt files found in {data_path}")
            return {"message": "No documents found", "documents": 0, "chunks": 0}

        documents = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    filename = os.path.basename(file_path)
                    documents.append({"text": content, "metadata": {"source": filename}})
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []
        metadatas = []
        for doc in documents:
            splits = text_splitter.split_text(doc["text"])
            chunks.extend(splits)
            metadatas.extend([doc["metadata"]] * len(splits))
        
        self.vector_store.add_texts(texts=chunks, metadatas=metadatas)
        self.vector_store.persist()
        
        logger.info("Ingestion completed successfully.")
        return {
            "message": "Documents ingested successfully",
            "documents": len(documents),
            "chunks": len(chunks)
        }

    def ask_question(self, question: str) -> dict:
        logger.info(f"Performing RAG retrieval for question: {question}")
        
        # Retrieve with score to calculate confidence
        results = self.vector_store.similarity_search_with_score(question, k=3)
        
        if not results:
            return {
                "answer": "I could not find this information in the provided documents.",
                "sources": [],
                "confidence": "low"
            }
            
        docs = [res[0] for res in results]
        
        # Calculate a naive similarity score based on Chroma's L2 distance
        # Lower L2 distance means higher similarity.
        # similarity = 1 / (1 + distance). Max is 1.0.
        best_distance = results[0][1]
        similarity_score = 1.0 / (1.0 + best_distance)
        
        if similarity_score > 0.8:
            confidence = "high"
        elif similarity_score >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
            
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        logger.info("Generating response from LLM...")
        try:
            response = self.llm.invoke(formatted_prompt).strip()
            
            if "I could not find this information" in response:
                confidence = "low"
                sources = []
            else:
                sources = [{"document": doc.metadata.get("source", "unknown"), "chunk": doc.page_content} for doc in docs]
                
            return {
                "answer": response,
                "sources": sources,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

_rag_service = None

def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
