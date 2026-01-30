from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline



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

print(indexation_pipeline_rag())

    
 

def pipeline_rag(query:str):

    embeding_model = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            )  
    
    #embed the question
    query = embeding_model.encode([query]).astype('float32')


    #reconnexion a ChromaDB
    vectorstore=chromadb(
        persist_directory = "rag/data/chroma_db",
        collection_name = "documents",
        embedding_function = embeding_model
    )

    #retriever
    retriever = vectorstore.as_retriever(
        search_type ="similarity",
        search_kwargs ={"k": 3}
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
{question}

RESPONSE
""",    
        input_variables =["context", "question"]
    )


    #LLM GENERATIVE RESPONSE
    hf_pipeline = pipeline(
        "text_generation",
        model = "google/flan-t5-base"
        
    )




  
    
    

