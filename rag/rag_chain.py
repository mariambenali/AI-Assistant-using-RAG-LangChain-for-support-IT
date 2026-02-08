from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import numpy as np
import mlflow
import os



os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



#configuration mlflow
mlflow.set_experiment("IT_Support_RAG_System")



def pipeline_rag(query:str):

    start_time= time.time()
    with mlflow.start_run():

        #params
        mlflow.log_param("llm_model", "mistral-7b")
        mlflow.log_param("temperature",0.2)
        mlflow.log_param("top_k",5)

        
        embeding_model = HuggingFaceEmbeddings(
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                )  

        #reconnexion a ChromaDB
        vectorstore=Chroma(
            persist_directory = "./chroma_db", 
            collection_name = "it_documents", 
            embedding_function = embeding_model 
        )

        #retriever
        retriever = vectorstore.as_retriever(
            search_type ="similarity",
            search_kwargs ={"k": 3}
        )

        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("chunk_size", 500)
        mlflow.log_param("retriever_k", 4)

        mlflow.log_param("chunk_size", 500)
        mlflow.log_param("retriever_k", 4)

        

        #Prompt
        prompt = PromptTemplate(
            template = """
    Tu es un assistant IT.
    Réponds uniquement à partir du CONTEXTE ci-dessous.
    Si la réponse n’est pas dans le contexte, dis clairement : "Je ne sais pas".

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
        
        
        # chain RAG
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)


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
        



if  __name__ == "__main__":

    print("Question : What is Windows NT?")
    response = pipeline_rag("what is Windows NT ?")
    print(f"Réponse : {response}")