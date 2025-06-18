# defined_prompt function
# Agent creation and orchestration
# Main conversation loop

import asyncio
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_store
from langsmith import traceable
from rich.console import Console

from .config import DEFAULT_CONFIG, validate_environment
from .tools import profileData, qdrant_search_memory_tool, create_memory_tools
from .services import database_service

console = Console()

async def defined_prompt(state, config):
    """Define the system prompt for the agent"""
    user_id = config['configurable']['user_id']
    thread_id = config['configurable']['thread_id']
    namespace = ("chat", user_id, thread_id)

    # Log the namespace used for memory search
    console.print(f"Namespace for memory search in defined_prompt: {namespace}")

    # Use asearch for vector similarity search (asynchronous method)
    store = get_store()
    if store is None:
        raise ValueError("Store not found in config. Ensure the store is passed to the agent correctly.")
    
    items = await store.asearch(namespace, query=state["messages"][-1].content)
    memories = "\n\n".join(str(item) for item in items)

    # Log the memories found
    console.print(f"Found memories: {memories if memories else 'None'}")

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

@traceable(run_type="llm", metadata={"component": "main_query_loop", "user_id": "dynamic"})
async def run_query():
    """Main function to run the query loop"""
    
    # Validate environment
    validate_environment()
    
    console.print("set env variable OPENAI_API_KEY without fail")
    
    # Extract user and thread configuration
    config = DEFAULT_CONFIG
    user_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]

    # Ensure config contains resolved values for user_id and thread_id
    config_with_resolved_ids = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id
        }
    }

    # Create memory tools
    search_memory_tool, manage_memory_tool = create_memory_tools(user_id, thread_id)
    
    # Get checkpointer and embedder
    memory_checkpointer = database_service.get_memory_checkpointer()
    embedder = database_service.get_embedder()

    # Setup async store
    async with await database_service.get_async_postgres_store() as store:
        # Set up the store with the embeddings index
        try:
            await store.setup()
            console.print("Successfully set up AsyncPostgresStore.")
        except Exception as e:
            console.print(f"[Postgres Error] Failed to set up AsyncPostgresStore: {str(e)}")
            raise

        # Create agent with memory tools
        main_agent = create_react_agent(
            tools=[
                profileData,
                search_memory_tool,
                qdrant_search_memory_tool,
                manage_memory_tool,
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
        console.print(f"Namespace for storage: {namespace}")

        # Interactive loop to prompt the user for queries
        while True:
            # Prompt the user for a query
            user_query = input("Please enter your query (or type 'quit' to quit): ").strip()

            # Check if the user wants to quit
            if user_query.lower() == "quit":
                console.print("Exiting the interactive session. Goodbye!")
                break

            # Skip empty queries
            if not user_query:
                console.print("Empty query. Please enter a valid question.")
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
                console.print("\nAgent Response:")
                console.print(response)

                # Extract the agent's response for display
                agent_response = response["messages"][-1].content
                console.print(f"\n[Answer]: {agent_response}\n")

            except Exception as e:
                console.print(f"[Agent Error] Failed to process query: {str(e)}")
                continue

            # Verify stored data by searching
            try:
                stored_items = await store.asearch(namespace, query=user_query)
                console.print("Stored items in database after query:")
                if stored_items:
                    for item in stored_items:
                        console.print(f"Item: {item.value}, Score: {item.score}")
                else:
                    console.print("No items found in the database for this query.")
            except Exception as e:
                console.print(f"[Postgres Error] Failed to search database after query: {str(e)}")
                continue