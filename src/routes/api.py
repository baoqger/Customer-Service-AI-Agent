from fastapi import APIRouter, HTTPException
from typing import List
from ..models import TaskItem, TaskCreateRequest, TaskUpdateRequest, ChatRequest, ChatMessage
from ..agents import SemanticKernelAgent


def create_api_routes(
    sk_agent: SemanticKernelAgent
) -> APIRouter:
    """
    Create API router with task CRUD endpoints and chat agent routes.
    
    Routes:
    - POST   /chat/semantic : Processes a chat message using the Semantic Kernel agent
    - POST   /chat/foundry   : Processes a chat message using the Foundry agent
    """
    router = APIRouter()
    
    @router.post("/chat/semantic", response_model=ChatMessage, operation_id="chatWithSK", include_in_schema=False)
    async def chat_with_foundry(chat_request: ChatRequest):
        """Process a chat message using the SK agent"""
        try:
            if not chat_request.message:
                raise HTTPException(status_code=400, detail="Message is required")
            print("process request in SK Agent")
            response = await sk_agent.process_message(chat_request.message)
            return response
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error in Semantic Kernel chat: {e}")
            raise HTTPException(status_code=500, detail="Failed to process message")
    
    return router
