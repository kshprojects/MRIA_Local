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
import json,ast,os
import datetime as dt
from rich.console import Console

from .models import UserProfile
from .services import qdrant_service, model_service, gcs_service, gemini_service
from .storage_utils import quick_background_save, create_timestamped_folder, start_background_image_storage, prepare_images_for_background_storage
from .config import (
    GAP_THRESHOLD,MIN_RESULTS,MAX_RESULTS
)

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

@traceable(run_type="tool", metadata={"tool_name": "LLM_filter", "purpose": "Categories Book Categories to search in vector database"})
def LLM_filter(query:str):
    """Categorizes the user query into any of the predefined Book_Categories, it can also be list of categories"""

    Book_Categories = ["Anesthesia", "Pain_Management", "Critical_Care", "Internal_Medicine","Dosing"]

    # Construct the prompt for categorization
    prompt_text = f"""
    You are an expert in medical literature categorization.
    Categorize the following query into one of the predefined Book_Categories: {', '.join(Book_Categories)}.

    User's Query:
    {query}

    Your task:
    - Identify the most relevant category based on the content of the query.
    - Return only the category names as a List. If uncertain, return empty list.
    - No preambles,No extra text, explanations, or formatting.
    """

    # Use the existing GeminiService from services.py
    try:
        print("Sending query to Gemini API for CATEGORIZATION...")
        
        # Create a simple images_payload (empty list since we're only using text)
        images_payload = []
        
        # Use the existing gemini_service.generate_content method
        response_text = gemini_service.generate_content(prompt_text, images_payload)
        
        if response_text is None:
            print("No response from Gemini API, returning None.")
            return None

        # print("API Response:", response_text)
        category_list = ast.literal_eval(response_text.strip())

        if isinstance(category_list, list):
            print("Categorized Response:", category_list)
            return category_list
        else:
            print("Invalid format received, returning None.")
            return None

    except Exception as e:
        print(f"Error occurred: {e}")
        return None  # Return None if category couldn't be determined
    

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
@traceable(run_type="retriever", metadata={"tool_name": "qdrant_search_memory_tool", "purpose": "Vector database search"})
def qdrant_search_memory_tool(query: str):
    """
    Processes a user query to retrieve relevant documents from a vector database
    and generates a response using a Large Language Model (LLM).
    """

    category_list = LLM_filter(query) 

    if isinstance(query, str):
        query = [query]

    try:
        colqwen_query = model_service.batch_embed_query(query)
    except Exception as e:
        return f"Error embedding query: {str(e)}"

    # Search Qdrant
    answer_colqwen = qdrant_service.reranking_search_batch(colqwen_query, filter_list=category_list)
    if isinstance(answer_colqwen, str):  # Check if an error message was returned
        return answer_colqwen

    #Apply gap filtering to pre-filtered results
    try:
        initial_points = answer_colqwen[0].points
        console.print(f"[Initial Results] Found {len(initial_points)} results above basic threshold")
        
        if not initial_points:
            return "No documents found above the score threshold."
    
        # Display all scores for analysis
        all_scores = [point.score for point in initial_points]
        console.print(f"[Score Distribution] {[round(score, 2) for score in all_scores[:10]]}")
        
        # Sort points by score (highest first) - should already be sorted but ensure it
        sorted_points = sorted(initial_points, key=lambda p: p.score, reverse=True)
        
        # Always keep the top result
        filtered_points = [sorted_points[0]]
        console.print(f"[Gap Filter] Rank 1: {sorted_points[0].score:.2f} (top result - always keep)")
        
        # Apply gap analysis starting from rank 2
        for i in range(1, len(sorted_points)):
            current_score = sorted_points[i].score
            previous_score = sorted_points[i-1].score
            gap = previous_score - current_score
            
            console.print(f"[Gap Filter] Rank {i+1}: {current_score:.2f} (gap: {gap:.2f})")
            
            # Check for significant gap
            if gap >= GAP_THRESHOLD:
                console.print(f"[Gap Filter] 🛑 SIGNIFICANT GAP: {gap:.2f} ≥ {GAP_THRESHOLD} - stopping here")
                console.print(f"[Gap Filter] Quality cluster contains ranks 1-{i} ({len(filtered_points)} results)")
                break
            
            # Small gap - add to same quality cluster
            filtered_points.append(sorted_points[i])
            console.print(f"[Gap Filter] ✅ Small gap ({gap:.2f}) - adding to quality cluster")
            
            # Respect max results limit
            if len(filtered_points) >= MAX_RESULTS:
                console.print(f"[Gap Filter] 📊 Reached max results limit: {MAX_RESULTS}")
                break
        
        # Ensure minimum results (fallback safety)
        if len(filtered_points) < MIN_RESULTS and len(sorted_points) >= MIN_RESULTS:
            console.print(f"[Gap Filter] 📋 Ensuring minimum {MIN_RESULTS} results")
            filtered_points = sorted_points[:MIN_RESULTS]
        
        # Log final selection
        final_scores = [round(p.score, 2) for p in filtered_points]
        console.print(f"[Gap Filter] 🎯 Final selection: {len(filtered_points)} results")
        console.print(f"[Gap Filter] Selected scores: {final_scores}")
        
        # Quality analysis
        if len(filtered_points) > 1:
            score_range = filtered_points[0].score - filtered_points[-1].score
            avg_score = sum(p.score for p in filtered_points) / len(filtered_points)
            console.print(f"[Quality Analysis] Score range: {score_range:.2f}, Average: {avg_score:.2f}")
        
        # Convert to final result format with scores included
        top_results = []
        for point in filtered_points:
            top_results.append({
                "image_link": point.payload['image_link'],
                "score": point.score  # Include score for logging/debugging
            })
        
        console.print(f"[Final Results] Processing {len(top_results)} gap-filtered results")
        
        # Optional: Show what was filtered out
        if len(initial_points) > len(filtered_points):
            filtered_out = len(initial_points) - len(filtered_points)
            lowest_kept = filtered_points[-1].score if filtered_points else 0
            console.print(f"[Gap Filter Impact] Filtered out {filtered_out} results below score {lowest_kept:.2f}")

    except Exception as e:
        return f"Error processing Qdrant results: {str(e)}"

    if not top_results:
        return "No documents found after gap filtering."

    # Continue with rest of processing (image download, etc.)
    console.print(f"[Processing] Continuing with {len(top_results)} high-confidence results...")


    # 1. Download images for API (synchronous - required for Gemini)
    console.print(f"[Main Flow] 📥 Downloading {len(top_results)} images for API...")
    
    images_payload = []
    image_downloads = []  # Store download data for background saving
    
    for i, result in enumerate(top_results):
        try:
            image_data = gcs_service.download_gcs_image(result["image_link"])
            if isinstance(image_data, str):  # Error message
                console.print(f"[Main Flow] ❌ Failed to download image {i+1}: {image_data}")
                continue
            
            # Add to Gemini API payload (main flow)
            images_payload.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=image_data
                    )
                )
            )
            
            # Store for background saving
            image_downloads.append({
                'image_data': image_data,
                'gcs_uri': result["image_link"],
                'score': result.get("score", 0.0)
            })
            
        except Exception as e:
            console.print(f"[Main Flow] ❌ Error processing image {i+1}: {e}")
            continue
    
    console.print(f"[Main Flow] ✅ Prepared {len(images_payload)} images for API")
    
    # 2. Start background image saving (non-blocking)
    if image_downloads:
        try:
            # Use the utility function for clean background storage
            storage_thread = quick_background_save(
                image_downloads, 
                base_path="assets/retrieved_images"
            )
            console.print(f"[Background] 🚀 Started saving {len(image_downloads)} images")
        except Exception as e:
            console.print(f"[Background] ❌ Failed to start background storage: {e}")
    
    # 3. Continue with main flow (Gemini API call)
    if not images_payload:
        return "Failed to download any images for analysis."

    console.print(f"[Main Flow] 🧠 Sending {len(images_payload)} images to Gemini API...")

    # Create prompt for Gemini
    prompt_text = f"""
    You are a highly knowledgeable assistant with expertise in analyzing and synthesizing information.
    Below are relevant details (images, metadata) to answer the user's question accurately.

    User's Question:
    {query}

    Your task:
    - Analyze the images provided.
    - Use the metadata to generate an accurate and detailed response.
    - Even if it is formulae for medical related content, you can print that formulae to the user.
    - Avoid unrelated or speculative information.
    - Ensure the response is clear, concise, directly addresses the user's query.
    - Dont add your own points to the answer.
    """

    # Generate response using Gemini
    gemini_response = gemini_service.generate_content(prompt_text, images_payload)
    return gemini_response

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