from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import chromadb
#from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import joblib
import numpy as np
from ml.kmeans import predict_cluster
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# load model Kmeans
kmeans_model = joblib.load("ml/kmeans_it_support.pkl")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


import mlflow

#configuration 
mlflow.set_experiment("IT_Support_RAG_System")

def pipeline_rag(query: str):
    with mlflow.start_run():

        # PARAMS 
        mlflow.log_param("query", query)
        mlflow.log_param("model_name", "flan-t5-base")


        # load model hugginfaceEmbedding
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Connect to chromaDb
        vectorstore = Chroma(
            persist_directory="./rag/data/chroma_db",
            collection_name="it_documents",
            embedding_function=embedding_model
        )

        # the predict function 
        cluster_id = predict_cluster(query) 

        # Retriever 
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,
                "filter": {"cluster_id": int(cluster_id)}
            }
        )


        #Prompt
        prompt = PromptTemplate(
            template = """
    You are assistant IT.
    Answer the following IT question using only the context provided.
    If you don't know, say 'I don't know'.

    CONTEXTE:
    {context}

    QUESTION:
    {input}

    RESPONSE
    """,    
            input_variables =["context", "input"]
        )

        #LLM GENERATIVE RESPONSE
        hf_pipeline = pipeline(
            "text2text-generation",
            model = "google/flan-t5-base",
            max_new_tokens = 200
        )
        

        llm = HuggingFacePipeline(pipeline = hf_pipeline)

        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)

        cluster_id= predict_cluster(query)
        mlflow.log_metric("predicted_cluster", cluster_id)

        # INFERENCE
        start_time = time.time()
        result = qa_chain.invoke({"input": query})
        latency = (time.time() - start_time) * 1000


        # METRICS
        mlflow.log_metric("latency_ms", latency)
        mlflow.log_text(result["answer"], "response.txt")

        nb_docs = len(result.get("context", []))
        mlflow.log_metric("nb_retrieved_docs", nb_docs)


        return result["answer"]