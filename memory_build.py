from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch,uuid
from tqdm import tqdm
import base64,os
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from typing import List, Optional
from pydantic import BaseModel, Field
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from google import genai
from google.genai import types
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import PIL, io, base64, numpy as np
from google.cloud import storage
from io import BytesIO
import PIL.Image
from langchain_core.messages import SystemMessage


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
    # Handle the case where the collection does not exist"
    if "Collection not found" in str(e):  # Check for the specific error message
      print(f"collection '{collection_name}' Not found")

print("Going to load the model!")
model_name = ("vidore/colqwen2-v0.1")

model_colqwen= ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
    ).eval()

processor = ColQwen2Processor.from_pretrained(model_name)


"""#Main#"""

profile_store = InMemoryStore()

print("At Profile")
"""User profile specifically designed for medical professionals"""

def profileData(query: str):
    """
        Extracts and manages detailed user profile information from natural language input,
        specifically tailored for healthcare professionals interacting with MRIAs (Medical Retrieval Intelligence Assistant).

        The tool intelligently identifies and updates structured user data including:
            - Basic personal details (name, age, gender, location, etc.)
            - Professional background (role, specialty, experience)
            - Clinical and research interests
            - Preferred tools and resources used in practice

        Process Flow:
        1. First, searches the existing profile store using 'create_search_memory_tool' to retrieve any previously stored profile.
        2. Analyzes the provided query or message to extract new or updated profile information.
        3. Only explicitly stated or strongly implied fields are populated; ambiguous or missing values are set to null.
        4. Updates or creates a new profile entry using 'create_manage_memory_tool'.

        This contextual profile is then made available to the main agent (MRIAs) to personalize responses,
        improve clinical relevance, and support adaptive decision-making.

        Parameters:
        ----------
        query : str
            A string containing the user's message or input that may include personal or professional details.

        Returns:
        -------
        UserProfile (BaseModel)
            A structured model containing the extracted or updated user profile data.
            If no new data is found, returns the existing profile or an empty/defaulted model.

        Notes:
        -----
        - Designed for use within a clinical assistant environment where context-awareness improves response quality.
        - Uses conservative extraction logic to avoid hallucinations or assumptions.
        - Profile data is stored under a unique namespace per user for persistence across sessions.
    """

    class UserProfile(BaseModel):
        # Basic Information
        name: Optional[str] = Field(default=None, description="The user's preferred name")
        age: Optional[int] = Field(default=None, ge=18, le=100, description="The user's age")
        gender: Optional[str] = Field(default=None, description="The user's gender")
        location: Optional[str] = Field(default=None, description="City/State/Country where the user practices")
        timezone: Optional[str] = Field(default=None, description="The user's timezone")
        language: Optional[str] = Field(default="English", description="Preferred language for communication")

        # Professional Information
        medical_role: Optional[str] = Field(default=None, description="Primary medical role/profession")
        specialty: Optional[str] = Field(default=None, description="Medical specialty or area of focus")
        subspecialty: Optional[str] = Field(default=None, description="Subspecialty within their field")
        years_of_experience: Optional[int] = Field(default=None, ge=0, description="Years of professional experience")

        # Professional Interests & Goals
        clinical_interests: Optional[List[str]] = Field(default=None, description="Specific clinical areas of interest")
        research_interests: Optional[List[str]] = Field(default=None, description="Research areas of interest")
        career_goals: Optional[List[str]] = Field(default=None, description="Professional career objectives")

        # Technology & Tools
        medical_software_used: Optional[List[str]] = Field(default=None, description="Electronic health records and medical software used")
        preferred_medical_resources: Optional[List[str]] = Field(default=None, description="Preferred medical references and resources")


    # Profile agent prompt
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

    namespace = ("1","profile")

    # Create agent with memory tools
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

    profile_agent.invoke({"messages": [{"role": "user", "content": query}]},config={"configurable": {"user_id": "1"}})

#print(data_store.search((user_id,"profile")))

#GEMINI
def qdrant_search_memory_tool(query: str):
    """
    Processes a user query to retrieve relevant documents from a vector database
    and generates a response using a Large Language Model (LLM).

    Parameters:
            query:str
    Steps:
    1. Extracts the latest user query from the state.
    2. Converts the query into vector embeddings using a model.
    3. Searches a Qdrant vector database with the embeddings to retrieve top results.
    4. Constructs an input message for an LLM using the retrieved results that has base64 images.
    5. Invokes the LLM to generate a detailed and accurate response.

    Returns:
        list: The updated list of messages with the LLM's response appended.
    """
    #query = state['messages'][-1]
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

    LLM_category=['Internal Medicine']

    # Search the Qdrant vector database
    def reranking_search_batch(query_batch,
                           collection_name,
                           search_limit=20,
                           prefetch_limit=200,
                           FilterList=LLM_category):
      filter_ = None

      # Apply filter only if FilterList is not empty
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

      # If no results, perform a broad search without filters
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
                  filter=None,  # Broad search
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
      """Download an image from GCS and return its raw bytes."""
      if not gcs_uri.startswith("gs://"):
          raise ValueError("Invalid GCS URI format")

      bucket_name = gcs_uri.split("/")[2]
      blob_path = "/".join(gcs_uri.split("/")[3:])

      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(blob_path)

      return blob.download_as_bytes()  # Return raw image bytes

    # Convert GCS images to inline data format
    images_payload = [
        types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=download_gcs_image(result["image_link"])
            )
        )
        for result in top_10_results
      ]

    # Construct the prompt
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

    # Send request to Gemini Vision API
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

data_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

def defined_prompt(state,config):
    messages = state['messages']
    user_id = config['configurable']['user_id']
    thread_id = config['configurable']['thread_id']

    if not messages:
        return None

    # Get the last 2 messages (or fewer if not enough)
    prev_msgs = messages[-3:]

    # Format previous messages for context
    formatted_prev_msgs = "\n".join(
        f"{msg.type.capitalize()}: {msg.content}" for msg in prev_msgs
    )

    #print("formatted_prev_msgs",formatted_prev_msgs)

    system_msg = f"""
              You are MRIAs, a clinically intelligent assistant designed to support healthcare professionals by retrieving and synthesizing medical information with precision. You operate at the intersection of user memory, vector database knowledge, and clinical expertise.

              Before responding to any query:
              1. Search the user's memory using the "search_memory_tool" to understand prior interactions, patient history, or context-specific details.
              2.For your context, previous two conversations with the user are provided as "prev_msgs" below. The user may ask to summarize, reformat, or reference these directly — handle accordingly.
              3. If memory search yields no useful result, fallback to "qdrant_search_memory_tool" to fetch relevant evidence-based data.
              4. Cross-check all responses against clinical best practices and guidelines.
              5. When uncertain, advise consultation with a licensed medical professional.

              When answering:
              - Be concise, clear, and grounded in evidence-based medicine.
              - Personalize responses using user or patient profiles via the "profileData" tool.
              - Maintain a professional tone appropriate for clinical settings.
              - Prioritize safety and accuracy over completeness.

              After generating your response:
              1. Use "create_manage_memory_tool" to store. both the user query and your reply (especially if informed by "qdrant_search_memory_tool") for future use.
              2. only relevant and structured information should be stored , not every single message or tool call.
              3. Your goal is to be a smart, safe, clinical bridge — ensuring each answer supports informed decisions aligned with the highest care standards.

              <Previous Context ("prev_msgs")>
              {formatted_prev_msgs}
              </Previous Context ("prev_msgs")>

          """
    #print("=== DEBUG: System Message ===")
    #print(system_msg)
    #print("=============================")

    # Return full prompt structure: system message first, then user/assistant messages
    return [SystemMessage(content=system_msg)] + messages


search_memory_tool = create_search_memory_tool(
  namespace=("chat", "{user_id}","{thread_id}"),
  instructions=f"""
                Use this tool to search for relevant prior information stored in the user's memory, such as medical history, medications, allergies, prior queries, or clinical context.
                Call this tool:
                1. At the beginning of each new user query to retrieve any related or previously stored information.
                2. When attempting to personalize a response based on past interactions.
                if this memory search does not yield any relevant results or gives out null, immediately fallback to the "qdrant_search_memory_tool" to retrieve context from the broader historical vector database.
        """
)

createmanage_memory_tool=create_manage_memory_tool(namespace=('chat',"{user_id}","{thread_id}"))


# Create agent with memory tools
main_agent = create_react_agent(
    model="openai:gpt-4o",
    #model="google_genai:gemini-1.5-pro",
    tools=[
        profileData,
        search_memory_tool,
        qdrant_search_memory_tool,
        createmanage_memory_tool,
    ],
    prompt=defined_prompt,
    store=data_store
)

config = {
    "configurable": {
        "user_id": "1",
        "thread_id":"1"
    }
}
user_query = input("Enter the user_query : ")

response = main_agent.invoke(
    {"messages": [
        {"role": "user", "content": user_query}]
    },config=config
)

print("AI response:",response)

# response = main_agent.invoke(
#     {"messages": [
#         {"role": "user", "content": "what are the PRINCIPLES OF SCREENING"}]
#     },config=config
# )

# print(response["messages"][-1].content)

#print(profile_store.search((user_id,"profile")))
#print(data_store.search(("chat", user_id)))



# """##POSTGRES##"""

# !pip install psycopg2-binary asyncio langgraph-checkpoint-postgres

# import asyncio
# from langchain.embeddings import init_embeddings
# from langgraph.store.postgres import AsyncPostgresStore

# conn_string = "postgresql://postgres:12345@localhost:5432/MRIA"

# async def create_postgres_store():
#     """Create and initialize PostgreSQL store"""

#     postgres_store = AsyncPostgresStore.from_conn_string(
#         conn_string,
#         index={
#             "dims": 1536,
#             "embed": init_embeddings("openai:text-embedding-3-small"),
#             "fields": ["content", "metadata"]  # Fields to embed for search
#         }
#     )

#     # Initialize database schema (run once)
#     async with postgres_store as s:
#         await s.setup()

#     return postgres_store

# store = await create_postgres_store()