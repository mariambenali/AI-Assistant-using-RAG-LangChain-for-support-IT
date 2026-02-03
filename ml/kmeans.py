from sentence_transformers import SentenceTransformer
import joblib

# Load embedding model 
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load KMeans
KMEANS_PATH = "ml/kmeans_it_support.pkl"
kmeans_model = joblib.load(KMEANS_PATH)


def predict_cluster(question: str) -> int:
    """
    Take a question and return its cluster_id
    """
    embedding = st_model.encode([question])   
    cluster_id = kmeans_model.predict(embedding)[0]
    return int(cluster_id)

#predict_cluster("How to troubleshoot a network issue?")