# Environment variables, database connections, model configs
# Move all hardcoded values from memoryBuild.py here

import os
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Environment validation
def validate_environment():
    """Validate all required environment variables are set"""
    
    # Verify LangSmith environment variables
    if not os.getenv("LANGCHAIN_API_KEY"):
        console.print("[LangSmith Error] LANGCHAIN_API_KEY environment variable not set. Please set it in your .env file.")
        exit(1)
    if not os.getenv("LANGCHAIN_PROJECT"):
        console.print("[LangSmith Warning] LANGCHAIN_PROJECT not set. Default project will be used.")

    # Validate GOOGLE_APPLICATION_CREDENTIALS
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        console.print("[GCS Error] GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please set it to the path of your Google Cloud service account key JSON file.")
        exit(1)

    # Validate OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[OPEN_API_KEY] environment variable not set. Please set it to the path")
        exit(1)

    console.print("✅ All environment variables validated successfully")

# Database configurations
POSTGRES_CONNECTION_STRING = "postgresql://neondb_owner:npg_sOwWAdz0XMv5@ep-floral-bird-a1rrkx6w-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

# Qdrant configurations
QDRANT_URL = "https://1afa2602-77bb-44f6-95cb-9d5c3b930184.ap-south-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.zsvJrYsG8j6o1CKaJdrJvYYXmQboRfzxBvokg4u1ORs"
QDRANT_COLLECTION_NAME = "MRIA_test1"
QDRANT_TIMEOUT = 300

# Model configurations
MODEL_NAME = "vidore/colqwen2-v0.1"

# Google Cloud configurations
GCP_CREDENTIALS_PATH = "/Users/saivignesh/Documents/MRIA/upheld-radar-459515-f3-577a557aa321.json"
GEMINI_API_KEY = "AIzaSyBfSbHEPbT3JB6WX9DImuKaUyGTmUekakw"

# User and thread configuration
DEFAULT_CONFIG = {
    "configurable": {
        "user_id": "1",
        "thread_id": "1"
    }
}

# Search configurations
SEARCH_LIMIT = 15
PREFETCH_LIMIT = 200
SCORE_THRESHOLD = 12

#gap threshold
GAP_THRESHOLD = 3.0
MIN_RESULTS = 2
MAX_RESULTS = SEARCH_LIMIT

ENABLE_BACKGROUND_STORAGE = True  
DEBUG_STORAGE = True  

# Set Google Cloud credentials environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH