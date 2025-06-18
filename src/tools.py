# profileData tool
# qdrant_search_memory_tool
# Memory management tools

from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from langsmith import traceable
from rich.console import Console

from .models import UserProfile
from .services import qdrant_service, model_service, gcs_service, gemini_service

console = Console()

@tool
@traceable(run_type="tool", metadata={"tool_name": "profileData", "purpose": "User profile extraction"})
def profileData(query: str, config):
    """
    Extracts and manages detailed user profile information from natural language input,
    specifically tailored for healthcare professionals interacting with MRIAs.
    """
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
    console.print(f"ProfileData - Namespace: {namespace}")
    console.print(f"ProfileData - Config: {config}")

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

    try:
        colqwen_query = model_service.batch_embed_query(query)
    except Exception as e:
        return f"Error embedding query: {str(e)}"

    # Search Qdrant
    answer_colqwen = qdrant_service.reranking_search_batch(colqwen_query)
    if isinstance(answer_colqwen, str):  # Check if an error message was returned
        return answer_colqwen

    # Extract top 10 results
    top_10_results = []
    try:
        for point in answer_colqwen[0].points[:10]:
            top_10_results.append({"image_link": point.payload['image_link']})
    except Exception as e:
        return f"Error processing Qdrant results: {str(e)}"

    # Download images from GCS
    images_payload = []
    for result in top_10_results:
        image_data = gcs_service.download_gcs_image(result["image_link"])
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

    # Create prompt for Gemini
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

    # Generate response using Gemini
    return gemini_service.generate_content(prompt_text, images_payload)

def create_memory_tools(user_id: str, thread_id: str):
    """Create memory tools for a specific user and thread"""
    namespace = ("chat", user_id, thread_id)
    
    search_memory_tool = create_search_memory_tool(
        namespace=namespace,
        instructions=f"""
            Use this tool to search for relevant prior information stored in the user's memory, such as medical history, medications, allergies, prior queries, or clinical context.
            Call this tool:
            1. At the beginning of each new user query to retrieve any related or previously stored information.
            2. When attempting to personalize a response based on past interactions.
            If this memory search does not yield any relevant results or gives out null, immediately fallback to the "qdrant_search_memory_tool" to retrieve context from the broader historical vector database.
        """
    )
    
    manage_memory_tool = create_manage_memory_tool(namespace=namespace)
    
    return search_memory_tool, manage_memory_tool