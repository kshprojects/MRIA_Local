# Imports
import os
import asyncio
import base64
import uuid
from typing import List, Optional
from io import BytesIO
import torch
import numpy as np
from tqdm import tqdm
import PIL
import json
from dotenv import load_dotenv
from rich.console import Console
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from google import genai
from google.genai import types
from langchain_core.tools import tool
from google.cloud import storage
from langchain.embeddings import init_embeddings
from langgraph.store.postgres.aio import AsyncPostgresStore  # Use only AsyncPostgresStore
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.config import get_store

# Initialize dotenv to load environment variables
load_dotenv()

# Initialize Rich for better output formatting and visualization
rich = Console()

# Provided configurations
conn_string = "postgresql://neondb_owner:npg_sOwWAdz0XMv5@ep-floral-bird-a1rrkx6w-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

print("set env variable OPENAI_API_KEY without fail")

qdrantclient = QdrantClient(
    url="https://1afa2602-77bb-44f6-95cb-9d5c3b930184.ap-south-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.zsvJrYsG8j6o1CKaJdrJvYYXmQboRfzxBvokg4u1ORs",
    timeout=100
)
collection_name = "MRIA_test1"

try:
    qdrantclient.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e:
    if "Collection not found" in str(e):
        print(f"collection '{collection_name}' Not found")

print("Going to load the model!")
model_name = "vidore/colqwen2-v0.1"

model_colqwen = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
).eval()

processor = ColQwen2Processor.from_pretrained(model_name, use_fast=True)  # Added use_fast=True

# Profile store and tools setup
profile_store = InMemoryStore()

print("At Profile")
"""User profile specifically designed for medical professionals"""

@tool
def profileData(query: str, config):
    """
    Extracts and manages detailed user profile information from natural language input,
    specifically tailored for healthcare professionals interacting with MRIAs.
    """
    class UserProfile(BaseModel):
        name: Optional[str] = Field(default=None, description="The user's preferred name")
        age: Optional[int] = Field(default=None, ge=18, le=100, description="The user's age")
        gender: Optional[str] = Field(default=None, description="The user's gender")
        location: Optional[str] = Field(default=None, description="City/State/Country where the user practices")
        timezone: Optional[str] = Field(default=None, description="The user's timezone")
        language: Optional[str] = Field(default="English", description="Preferred language for communication")
        medical_role: Optional[str] = Field(default=None, description="Primary medical role/profession")
        specialty: Optional[str] = Field(default=None, description="Medical specialty or area of focus")
        subspecialty: Optional[str] = Field(default=None, description="Subspecialty within their field")
        years_of_experience: Optional[int] = Field(default=None, ge=0, description="Years of professional experience")
        clinical_interests: Optional[List[str]] = Field(default=None, description="Specific clinical areas of interest")
        research_interests: Optional[List[str]] = Field(default=None, description="Research areas of interest")
        career_goals: Optional[List[str]] = Field(default=None, description="Professional career objectives")
        medical_software_used: Optional[List[str]] = Field(default=None, description="Electronic health records and medical software used")
        preferred_medical_resources: Optional[List[str]] = Field(default=None, description="Preferred medical references and resources")

    profile_agent_prompt = f"""
    You are an assistant that extracts user profile information from messages.

    First, use the "create_search_memory_tool" to retrieve any personal profile information of the user.
    then use the tool "create_manage_memory_tool" to create a new profile or update the existing profile.

    Only extract values that are clearly stated or strongly implied.
    If any information is missing or unclear, leave the field as null.
    Do not guess or hallucinate missing values.

    Return the structured output using the given schema.

    Available profile fields to extract:
    - Basic Information: name, age, gender, location, timezone, language
    - Professional Information: medical_role, specialty, subspecialty, years_of_experience
    - Professional Interests: clinical_interests, research_interests, career_goals
    - Technology & Tools: medical_software_used, preferred_medical_resources

    Process:
    1. Search for existing profile using "create_search_memory_tool"
    2. Analyze the user's message for new profile information
    3. Extract only clearly stated or strongly implied information
    4. Use "create_manage_memory_tool" to save/update the profile
    5. Be conservative in extraction - only include explicit or clearly implied information

    Always maintain data accuracy and avoid assumptions.
    """

    user_id = config['configurable']['user_id']
    namespace = ("profile", user_id)

    profile_agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[
            create_search_memory_tool(namespace=namespace),
            create_manage_memory_tool(namespace=namespace),
        ],
        store=profile_store,
        prompt=profile_agent_prompt,
        response_format=UserProfile
    )

    return profile_agent.invoke({"messages": [{"role": "user", "content": query}]}, config={"configurable": {"user_id": user_id}})

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
def qdrant_search_memory_tool(query: str):
    """
    Processes a user query to retrieve relevant documents from a vector database
    and generates a response using a Large Language Model (LLM).
    """
    if isinstance(query, str):
        query = [query]

    def batch_embed_query(query_batch, model_processor, model):
        with torch.no_grad():
            processed_queries = model_processor.process_queries(query_batch).to(model.device)
            query_embeddings_batch = model(**processed_queries)
        return query_embeddings_batch.cpu().float().numpy()

    try:
        colqwen_query = batch_embed_query(query, processor, model_colqwen)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

    LLM_category = ['Internal Medicine']

    def reranking_search_batch(query_batch, collection_name, search_limit=20, prefetch_limit=200, FilterList=LLM_category):
        filter_ = None
        if FilterList:
            filter_ = models.Filter(
                should=[
                    models.FieldCondition(
                        key="Book_Category",
                        match=models.MatchAny(any=FilterList)
                    )
                ]
            )

        search_queries = [
            models.QueryRequest(
                query=query,
                prefetch=[
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_columns"
                    ),
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_rows"
                    ),
                ],
                filter=filter_,
                limit=search_limit,
                with_payload=True,
                with_vector=False,
                using="original"
            ) for query in query_batch
        ]

        response = qdrantclient.query_batch_points(
            collection_name=collection_name,
            requests=search_queries
        )

        if all(not res.points for res in response):
            search_queries = [
                models.QueryRequest(
                    query=query,
                    prefetch=[
                        models.Prefetch(
                            query=query,
                            limit=prefetch_limit,
                            using="mean_pooling_columns"
                        ),
                        models.Prefetch(
                            query=query,
                            limit=prefetch_limit,
                            using="mean_pooling_rows"
                        ),
                    ],
                    filter=None,
                    limit=search_limit,
                    with_payload=True,
                    with_vector=False,
                    using="original"
                ) for query in query_batch
            ]
            response = qdrantclient.query_batch_points(
                collection_name=collection_name,
                requests=search_queries
            )

        return response

    answer_colqwen = reranking_search_batch(colqwen_query, collection_name)

    top_10_results = []
    for point in answer_colqwen[0].points[:10]:
        top_10_results.append({"image_link": point.payload['image_link']})

    storage_client = storage.Client()

    def download_gcs_image(gcs_uri):
        if not gcs_uri.startswith("gs://"):
            raise ValueError("Invalid GCS URI format")
        bucket_name = gcs_uri.split("/")[2]
        blob_path = "/".join(gcs_uri.split("/")[3:])
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_bytes()

    images_payload = [
        types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=download_gcs_image(result["image_link"])
            )
        )
        for result in top_10_results
    ]

    prompt_text = f"""
    You are a highly knowledgeable assistant with expertise in analyzing and synthesizing information.
    Below are relevant details (images, metadata) to answer the user's question accurately.

    User's Question:
    {query}

    Your task:
    - Analyze the images provided.
    - Use the metadata to generate an accurate and detailed response.
    - Avoid unrelated or speculative information.
    - Ensure the response is clear, concise, directly addresses the user's query.
    - Dont add your own points to the answer.
    """

    try:
        client = genai.GenerativeModel("gemini-1.5-flash")  # Updated to use GenerativeModel
        print("Sending data to Gemini Vision API...")
        response = client.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt_text}] + images_payload}]
        )
        return response.text
    except Exception as e:
        print(f"GEMINI Error: {e}")
        return None

async def defined_prompt(state, config):
    user_id = config['configurable']['user_id']
    thread_id = config['configurable']['thread_id']
    namespace = ("chat", user_id, thread_id)

    # Use asearch for vector similarity search (asynchronous method)
    store = get_store()
    if store is None:
        raise ValueError("Store not found in config. Ensure the store is passed to the agent correctly.")
    
    items = await store.asearch(namespace, query=state["messages"][-1].content)
    memories = "\n\n".join(str(item) for item in items)

    system_msg = f"""
        You are MRIAs, a clinically intelligent assistant designed to support healthcare professionals by retrieving and synthesizing medical information with precision. You operate at the intersection of user memory, vector database knowledge, and clinical expertise.

        Before responding to any query:
        1. Search the user's memory using the `search_memory_tool` to understand prior interactions, patient history, or context-specific details.
        2. For your context, previous conversations with the user are provided as `memories` below. The user may ask to summarize, reformat, or reference these directly — handle accordingly.
        3. If memory search yields no useful result, fallback to `qdrant_search_memory_tool` to fetch relevant evidence-based data.
        4. Cross-check all responses against clinical best practices and guidelines.
        5. When uncertain, advise consultation with a licensed medical professional.

        When answering:
        - Be concise, clear, and grounded in evidence-based medicine.
        - Personalize responses using user or patient profiles via the `profileData` tool.
        - Maintain a professional tone appropriate for clinical settings.
        - Prioritize safety and accuracy over completeness.

        After generating your response:
        1. Use `create_manage_memory_tool` to store both the user query and your reply (especially if informed by `qdrant_search_memory_tool`) for future use.
        2. Only relevant and structured information should be stored, not every single message or tool call.
        3. Your goal is to be a smart, safe, clinical bridge — ensuring each answer supports informed decisions aligned with the highest care standards.

        ##Memories:\n\n{memories}
    """
    return [SystemMessage(content=system_msg)] + state["messages"]

# Define user and thread configuration
config = {
    "configurable": {
        "user_id": "1",
        "thread_id": "1"
    }
}

# Define memory tools
search_memory_tool = create_search_memory_tool(
    namespace=("chat", "{user_id}", "{thread_id}"),
    instructions=f"""
        Use this tool to search for relevant prior information stored in the user's memory, such as medical history, medications, allergies, prior queries, or clinical context.
        Call this tool:
        1. At the beginning of each new user query to retrieve any related or previously stored information.
        2. When attempting to personalize a response based on past interactions.
        If this memory search does not yield any relevant results or gives out null, immediately fallback to the "qdrant_search_memory_tool" to retrieve context from the broader historical vector database.
    """
)

createmanage_memory_tool = create_manage_memory_tool(namespace=('chat', "{user_id}", "{thread_id}"))
memory_checkpointer = MemorySaver()

# Initialize embeddings
embedder = init_embeddings("openai:text-embedding-3-small")

# Wrap invocation in an async main function
async def run_query():
    async with AsyncPostgresStore.from_conn_string(conn_string) as store:
        # Set up the store with the embeddings index
        await store.setup()

        # Create agent with memory tools
        main_agent = create_react_agent(
            tools=[
                profileData,
                search_memory_tool,
                qdrant_search_memory_tool,
                createmanage_memory_tool,
            ],
            prompt=defined_prompt,
            model="openai:gpt-4o",
            checkpointer=memory_checkpointer,
            store=store
        )

        # Update config to include the store
        config_with_store = {
            **config,
            "store": store
        }

        # Run the query
        response = await main_agent.ainvoke(
            {"messages": [HumanMessage(content="Give me the subheadings of the INDICATIONS FOR CARDIAC CATHETERIZATION AND CORONARY ANGIOGRAPHY")]},
            config=config_with_store
        )
        rich.print(response)

# Run the async function
if __name__ == "__main__":
    if os.name == "nt":  # For Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_query())