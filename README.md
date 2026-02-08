# AI-Assistant-using-RAG-LangChain-for-support-IT
un assistant intelligent interne capable de répondre de manière fiable aux questions des techniciens IT à partir d’un PDF de support IT (procédures, incidents, FAQ).




## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Déploiement](#-déploiement)
- [CI/CD](#-cicd)
- [Monitoring & MLOps](#-monitoring--mlops)
- [API Documentation](#-api-documentation)
- [Structure du Projet](#-structure-du-projet)
- [Contributing](#-contributing)

---

## 🎯 Vue d'ensemble

**RAG IT Assistant** est un assistant intelligent interne conçu pour améliorer l'efficacité des équipes support IT. Il utilise la technologie RAG (Retrieval-Augmented Generation) pour répondre de manière fiable aux questions des techniciens à partir d'une base de connaissance PDF.

### Objectifs du Projet

- ✅ Répondre rapidement aux questions récurrentes
- ✅ Guider les techniciens lors d'incidents
- ✅ Standardiser les procédures IT
- ✅ Assurer une traçabilité complète des interactions
- ✅ Permettre une amélioration continue via ML

### Points Clés

- 🔄 **Pipeline RAG** complet avec LangChain
- 🗄️ **Vector Database** avec ChromaDB
- 🔐 **API sécurisée** avec authentification JWT
- 📊 **MLflow** pour le tracking et model registry
- 🤖 **Clustering automatique** des questions utilisateurs
- 🚀 **CI/CD** avec GitHub Actions
- ☸️ **Déploiement Kubernetes** production-ready

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (JWT)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ /auth/login  │  │    /query    │  │   /history   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────┬────────────────────┬────────────────────┬───────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│  PostgreSQL  │    │  RAG Pipeline  │    │    MLflow    │
│   Database   │    │   (LangChain)  │    │   Tracking   │
│              │    │                │    │              │
│ - users      │    │ ┌────────────┐ │    │ - Runs       │
│ - queries    │    │ │ ChromaDB   │ │    │ - Models     │
│              │    │ │ (Vectors)  │ │    │ - Metrics    │
└──────────────┘    │ └────────────┘ │    └──────────────┘
                    │ ┌────────────┐ │
                    │ │ HuggingFace│ │
                    │ │ Embeddings │ │
                    │ └────────────┘ │
                    │ ┌────────────┐ │
                    │ │   Gemini   │ │
                    │ │    LLM     │ │
                    │ └────────────┘ │
                    └────────────────┘
```

### Composants Principaux

1. **PDF Ingestion Pipeline**
   - Extraction avec PyPDFLoader
   - Chunking intelligent avec métadonnées
   - Génération d'embeddings HuggingFace

2. **Vector Database (ChromaDB)**
   - Stockage persistant des embeddings
   - Recherche sémantique optimisée
   - Versionnement de la base

3. **RAG Pipeline (LangChain)**
   - Retriever sémantique
   - Prompt engineering contrôlé
   - RetrievalQA chain

4. **Backend FastAPI**
   - API REST sécurisée (JWT)
   - Gestion utilisateurs
   - Historique des requêtes

5. **PostgreSQL Database**
   - Authentification utilisateurs
   - Traçabilité des interactions
   - Support du clustering

6. **ML Pipeline (Non-supervisé)**
   - Clustering KMeans des questions
   - Analyse des sujets fréquents
   - Amélioration continue

7. **MLOps (MLflow)**
   - Tracking des expérimentations
   - Model Registry
   - Versionnement des pipelines

---

## ✨ Fonctionnalités

### 🔍 Recherche Sémantique
- Compréhension du contexte des questions IT
- Réponses basées sur le PDF de support
- Traçabilité des sources (numéros de page)

### 🔐 Authentification & Sécurité
- JWT tokens pour l'authentification
- Gestion des utilisateurs avec PostgreSQL
- Contrôle d'accès par utilisateur

### 📊 Analytics & Monitoring
- Historique complet des questions/réponses
- Métriques de latence en temps réel
- Clustering automatique des questions similaires

### 🔄 MLOps & Versioning
- Tracking MLflow de tous les runs
- Model Registry pour versionnement
- Reproductibilité garantie

### 🚀 Production-Ready
- CI/CD automatisé avec GitHub Actions
- Déploiement Kubernetes


---

## 🛠️ Prérequis

### Logiciels Requis

```bash
# Versions minimales
Python >= 3.9
Docker >= 20.10
Docker Compose >= 2.0
Kubernetes >= 1.24 (via Lens Desktop)
Git >= 2.30
```

### Services Externes

- **HuggingFace Account** (pour embeddings)
- **Google Cloud Account** (pour Gemini API) OU HuggingFace pour LLM open-source
- **MLflow Server** (local ou distant)

### Dépendances Python

Voir `requirements.txt` pour la liste complète.

---

## 📥 Installation

### 1. Cloner le Repository

```bash
git clone https://github.com/your-org/rag-it-assistant.git
cd rag-it-assistant
```

### 2. Créer l'Environnement Virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration des Variables d'Environnement

```bash
cp .env.example .env
# Éditer .env avec vos configurations
```

---

## ⚙️ Configuration

### Fichier `.env`

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rag_assistant

# JWT Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256




```

### Configuration du PDF

Placez votre PDF de support IT dans :

```
data/raw/support_it.pdf
```

---

## 🚀 Utilisation

### Mode Développement Local

#### 1. Démarrer les Services (Docker Compose)

```bash
docker-compose up -d
```

Services démarrés :
- PostgreSQL : `localhost:5432`
- MLflow : `localhost:5000`
- FastAPI : `localhost:8000`

#### 2. Ingestion du PDF

```bash
python scripts/ingest_pdf.py --pdf data/raw/support_it.pdf
```

#### 3. Démarrer le Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 4. Accéder à l'API

- **Documentation Swagger** : http://localhost:8000/docs
- **Documentation ReDoc** : http://localhost:8000/redoc
- **MLflow UI** : http://localhost:5000

### Utilisation de l'API

#### Authentification

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

Réponse :
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### Poser une Question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment réinitialiser un mot de passe Active Directory?"}'
```

Réponse :
```json
{
  "question": "Comment réinitialiser un mot de passe Active Directory?",
  "answer": "Pour réinitialiser un mot de passe AD...",
  "latency_ms": 245,
}
```

#### Consulter l'Historique

```bash
curl -X GET "http://localhost:8000/history?limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```


---

## 📊 Monitoring & MLOps

### MLflow Tracking

Chaque requête RAG est trackée avec :

```python
with mlflow.start_run():
    mlflow.log_params({
        "llm_model": "gemini-pro",
        "temperature": 0.7,
        "chunk_size": 500,
        "top_k": 5
    })
    
    mlflow.log_metrics({
        "latency_ms": 245,
        "similarity_score": 0.89
    })
    
    mlflow.log_artifacts("outputs/")
```

### Visualiser dans MLflow UI

```bash
mlflow ui --port 5000
```

Accédez à : http://localhost:5000

### Model Registry

```bash
# Enregistrer un nouveau modèle
python scripts/register_model.py --model-path models/rag_pipeline_v1
```

---

## 🐳 Déploiement

### Docker Build

```bash
docker build -t rag-it-assistant:latest .
```



---

## 🔄 CI/CD

### GitHub Actions Pipeline

Le fichier `.github/workflows/ci-cd.yml` automatise :

1. **Linting** (optionnel)
   - Flake8 pour style Python
   - Black pour formatage

2. **Tests**
   - Tests unitaires (pytest)
   - Tests d'intégration
   - Coverage > 80%

3. **Build Docker**
   - Construction de l'image
   - Push vers Docker Hub / Registry

4. **Déploiement**
   - Déploiement automatique sur K8s
   - Smoke tests post-déploiement

### Triggers

```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```


---

## 📚 API Documentation

### Endpoints Principaux

| Méthode | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| POST | `/auth/register` | Création compte |✅ |
| POST | `/auth/login` | Authentification utilisateur | ✅ |
| POST | `/query` | Poser une question au RAG | ✅ |

### Schémas de Données

#### QueryRequest

```json
{
  "question": "string",
}
```

#### QueryResponse

```json
{
  "answer": "string",
  "latency_ms": "integer",
}
```

---

## 📁 Structure du Projet

```
rag-it-assistant/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           
├── app/
│   ├── database.py/
│   ├── main.py/
│   ├── models.py/             
│   ├── schema.py/        
│   └── security.py/             
├── data/
│   └── data_pdf/              
├── chromadb/            
├── ml/
│   ├── kmeans_it_support.pkl
│   ├── kmeans.py
│   ├── data_knowledge
│   └── ml_kmeans.ipynb
│   
├── rag/
│   ├── data          
│   ├── main.ipynb   
│   └── rag_chain.py        
│            
├── tests/
│   └── pipeline_test.py
│   
├── docker-compose.yml         
├── Dockerfile                 
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🤝 Contributing

Les contributions sont les bienvenues ! 

---

## 👥 Auteurs

- **Mariam BENALI** 
- 📧 Email : miriam.bena@gmail.com

---

**Made with ❤️ **