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
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.config import get_store
import rich
from langsmith import traceable


# Initialize dotenv to load environment variables
load_dotenv()

# Validate GOOGLE_APPLICATION_CREDENTIALS
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    rich.print("[GCS Error] GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please set it to the path of your Google Cloud service account key JSON file.")
    exit(1)

# Initialize Rich for better output formatting and visualization
rich = Console()

# Provided configurations
conn_string = "postgresql://neondb_owner:npg_sOwWAdz0XMv5@ep-floral-bird-a1rrkx6w-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

print("set env variable OPENAI_API_KEY without fail")


qdrantclient = QdrantClient(
    url="https://1afa2602-77bb-44f6-95cb-9d5c3b930184.ap-south-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.zsvJrYsG8j6o1CKaJdrJvYYXmQboRfzxBvokg4u1ORs",
    timeout=300
)
collection_name = "MRIA_test1"

try:
    qdrantclient.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e:
    if "Collection not found" in str(e):
        print(f"collection '{collection_name}' Not found")
    else:
        print(f"Error accessing Qdrant collection: {e}")

print("Going to load the model!")
model_name = "vidore/colqwen2-v0.1"

model_colqwen = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
).eval()

processor = ColQwen2Processor.from_pretrained(model_name, use_fast=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/saivignesh/Documents/MRIA/upheld-radar-459515-f3-577a557aa321.json"

# Profile store and tools setup
profile_store = InMemoryStore()

print("At Profile")
"""User profile specifically designed for medical professionals"""

@tool
@traceable(run_type="tool", metadata={"tool_name": "profileData", "purpose": "User profile extraction"})
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
    
    # Log the namespace and config for debugging
    rich.print(f"ProfileData - Namespace: {namespace}")
    rich.print(f"ProfileData - Config: {config}")

    # Create a separate InMemoryStore for profile_agent to avoid interference
    profile_agent_store = InMemoryStore()

    profile_agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[
            create_search_memory_tool(namespace=namespace),
            create_manage_memory_tool(namespace=namespace),
        ],
        store=profile_agent_store,  # Use a separate store
        prompt=profile_agent_prompt,
        response_format=UserProfile
    )

    # Ensure the config passed to profile_agent.invoke has resolved values
    profile_config = {
        "configurable": {
            "user_id": user_id
        }
    }

    return profile_agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=profile_config
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
@traceable(run_type="retriever", metadata={"tool_name": "qdrant_search_memory_tool", "purpose": "Vector database search"})
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
        return f"Error embedding query: {str(e)}"

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

        try:
            response = qdrantclient.query_batch_points(
                collection_name=collection_name,
                requests=search_queries
            )
        except Exception as e:
            rich.print(f"[Qdrant Error] Failed to query Qdrant with filter: {str(e)}")
            return f"Error querying Qdrant: {str(e)}"

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
            try:
                response = qdrantclient.query_batch_points(
                    collection_name=collection_name,
                    requests=search_queries
                )
            except Exception as e:
                rich.print(f"[Qdrant Error] Failed to query Qdrant (fallback): {str(e)}")
                return f"Error querying Qdrant (fallback): {str(e)}"

        return response

    answer_colqwen = reranking_search_batch(colqwen_query, collection_name)
    if isinstance(answer_colqwen, str):  # Check if an error message was returned
        return answer_colqwen

    top_10_results = []
    try:
        for point in answer_colqwen[0].points[:10]:
            top_10_results.append({"image_link": point.payload['image_link']})
    except Exception as e:
        return f"Error processing Qdrant results: {str(e)}"

    storage_client = storage.Client()

    def download_gcs_image(gcs_uri):
        try:
            if not gcs_uri.startswith("gs://"):
                raise ValueError("Invalid GCS URI format")
            bucket_name = gcs_uri.split("/")[2]
            blob_path = "/".join(gcs_uri.split("/")[3:])
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            return blob.download_as_bytes()
        except Exception as e:
            return f"Error downloading GCS image: {str(e)}"

    images_payload = []
    for result in top_10_results:
        image_data = download_gcs_image(result["image_link"])
        if isinstance(image_data, str):  # Check if an error message was returned
            return image_data
        images_payload.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=image_data
                )
            )
        )

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
        client = genai.Client(api_key="AIzaSyBfSbHEPbT3JB6WX9DImuKaUyGTmUekakw")
        storage_client = storage.Client()

        print("Sending data to Gemini Vision API...")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt_text)] + images_payload)]
        )
        return response.text
    except Exception as e:
        print(f"GEMINI Error: {e}")
        return None

async def defined_prompt(state, config):
    user_id = config['configurable']['user_id']
    thread_id = config['configurable']['thread_id']
    namespace = ("chat", user_id, thread_id)

    # Log the namespace used for memory search
    rich.print(f"Namespace for memory search in define d_prompt: {namespace}")

    # Use asearch for vector similarity search (asynchronous method)
    store = get_store()
    if store is None:
        raise ValueError("Store not found in config. Ensure the store is passed to the agent correctly.")
    
    items = await store.asearch(namespace, query=state["messages"][-1].content)
    memories = "\n\n".join(str(item) for item in items)

    # Log the memories found
    rich.print(f"Found memories: {memories if memories else 'None'}")

    system_msg = f"""
        You are MRIAs, a clinically intelligent assistant designed to support healthcare professionals by retrieving and synthesizing medical information with precision. You operate at the intersection of user memory, vector database knowledge, and clinical expertise.

        **Before responding to any query, you MUST follow these steps:**
        1. **ALWAYS** use the `search_memory_tool` to search the user's memory for prior interactions, patient history, or context-specific details. This step is mandatory, even if you think you know the answer.
        2. For your context, previous conversations with the user are provided as `memories` below. The user may ask to summarize, reformat, or reference these directly — handle accordingly.
        3. If the `search_memory_tool` yields no useful result (i.e., returns an empty list or null), **you MUST** use the `qdrant_search_memory_tool` to fetch relevant evidence-based data from the vector database. This step is also mandatory.
        4. Cross-check all responses against clinical best practices and guidelines.
        5. When uncertain, advise consultation with a licensed medical professional.

        **When answering:**
        - Be concise, clear, and grounded in evidence-based medicine.
        - Personalize responses using user or patient profiles via the `profileData` tool if relevant.
        - Maintain a professional tone appropriate for clinical settings.
        - Prioritize safety and accuracy over completeness.
        - Do not rely solely on your internal knowledge for medical queries. You must use the tools as instructed above.

        **After generating your response, you MUST follow these steps:**
        1. **ALWAYS** use `create_manage_memory_tool` to store both the user query and your reply in the database, regardless of whether the response was informed by `qdrant_search_memory_tool` or your internal knowledge. This step is mandatory.
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

# Explicitly extract user_id and thread_id
user_id = config["configurable"]["user_id"]
thread_id = config["configurable"]["thread_id"]

# Ensure config contains resolved values for user_id and thread_id
config_with_resolved_ids = {
    "configurable": {
        "user_id": user_id,
        "thread_id": thread_id
    }
}

# Define memory tools with explicit namespace values
search_memory_tool = create_search_memory_tool(
    namespace=("chat", user_id, thread_id),
    instructions=f"""
        Use this tool to search for relevant prior information stored in the user's memory, such as medical history, medications, allergies, prior queries, or clinical context.
        Call this tool:
        1. At the beginning of each new user query to retrieve any related or previously stored information.
        2. When attempting to personalize a response based on past interactions.
        If this memory search does not yield any relevant results or gives out null, immediately fallback to the "qdrant_search_memory_tool" to retrieve context from the broader historical vector database.
    """
)

createmanage_memory_tool = create_manage_memory_tool(
    namespace=("chat", user_id, thread_id)
)

memory_checkpointer = MemorySaver()

# Initialize embeddings
embedder = init_embeddings("openai:text-embedding-3-small")

# Wrap invocation in an async main function
@traceable(run_type="llm", metadata={"component": "main_query_loop", "user_id": "dynamic"})
async def run_query():
    async with AsyncPostgresStore.from_conn_string(conn_string) as store:
        # Set up the store with the embeddings index
        try:
            await store.setup()
            rich.print("Successfully set up AsyncPostgresStore.")
        except Exception as e:
            rich.print(f"[Postgres Error] Failed to set up AsyncPostgresStore: {str(e)}")
            raise

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

        # Update config to include the store and ensure resolved IDs
        config_with_store = {
            **config_with_resolved_ids,
            "store": store
        }

        namespace = ("chat", user_id, thread_id)
        rich.print(f"Namespace for storage: {namespace}")

        # Interactive loop to prompt the user for queries
        while True:
            # Prompt the user for a query
            user_query = input("Please enter your query (or type 'quit' to quit): ").strip()

            # Check if the user wants to quit
            if user_query.lower() == "quit":
                rich.print("Exiting the interactive session. Goodbye!")
                break

            # Skip empty queries
            if not user_query:
                rich.print("Empty query. Please enter a valid question.")
                continue

            # Create a HumanMessage with the user's query
            user_message = HumanMessage(content=user_query)

            # Embed the user query (for potential future use, e.g., searching the database)
            user_embedding = embedder.embed_documents([user_query])[0]

            # Run the agent, which will handle storing the conversation via create_manage_memory_tool
            try:
                response = await main_agent.ainvoke(
                    {"messages": [user_message]},
                    config=config_with_store
                )
                rich.print("\nAgent Response:")
                rich.print(response)

                # Extract the agent's response for display
                agent_response = response["messages"][-1].content
                rich.print(f"\n[Answer]: {agent_response}\n")

            except Exception as e:
                rich.print(f"[Agent Error] Failed to process query: {str(e)}")
                continue

            # Verify stored data by searching
            try:
                stored_items = await store.asearch(namespace, query=user_query)
                rich.print("Stored items in database after query:")
                if stored_items:
                    for item in stored_items:
                        rich.print(f"Item: {item.value}, Score: {item.score}")
                else:
                    rich.print("No items found in the database for this query.")
            except Exception as e:
                rich.print(f"[Postgres Error] Failed to search database after query: {str(e)}")
                continue

# Run the async function
if __name__ == "__main__":
    if os.name == "nt":  # For Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_query())