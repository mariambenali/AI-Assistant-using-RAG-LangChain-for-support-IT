from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import joblib
from ml.kmeans_utils import predict_cluster


def indexation_pipeline_rag():

    #load data
    loader = PyPDFLoader("/Users/miriambenali/Desktop/Project-Simplon/AI-Assistant-using-RAG-LangChain-for-support-IT/data/data-697729d43c37a040070748.pdf")
    documents = loader.load()

    #Chunk
    splitter =RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap= 50
)
    chunks = splitter.split_documents(documents) 

    #embedding
    embedding_model =HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            )  

    # Texte des chunks
    texts = [chunk.page_content for chunk in chunks]

    # Génération des embeddings 
    embeddings = embedding_model.embed_documents(texts) 


    #vectorizationDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="it_documents")

    collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(texts))]
    )

    return collection

#print(indexation_pipeline_rag())



def pipeline_rag(query:str):

    embeding_model = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            )  
    
    #reconnexion a ChromaDB
    vectorstore=Chroma(
        persist_directory = "./chroma_db",
        collection_name = "it_documents",
        embedding_function = embeding_model
    )

    #Predict cluster
    cluster_id = predict_cluster(query)

    #retriever
    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "filter": {"cluster_id": cluster_id}
    }
)

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

    result = qa_chain.invoke({"input": query})

    return result["answer"]



if  __name__ == "__main__":

    print("Question : What is Windows NT?")
    response = pipeline_rag("what is Windows NT ?")
    print(f"Réponse : {response}")