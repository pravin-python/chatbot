from fastapi import APIRouter, HTTPException, status
from app.models.chat import ChatRequest, ChatResponse, UserChatsResponse
from app.services.cache_service import cache_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post("/send", response_model=ChatResponse)
async def store_chat(payload: ChatRequest):
    """Store a chat message with auto-generated response"""
    try:
        # Main function call - process message and generate response'
        result = await cache_service.process_and_store_chat(
            user_id=payload.user_id,
            message=payload.message,
            product_id=payload.product_id
        )
        
        if result["success"]:
            return ChatResponse(
                success=True,
                message=f"Chat stored successfully. Total messages: {result.get('total_messages', 0)}",
                data=[result["chat"]] if "chat" in result else None
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to store chat")
            )
            
    except Exception as e:
        logger.error(f"Error in store_chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/data/{user_id}/{product_id}", response_model=UserChatsResponse)
async def get_user_chats(user_id: int, product_id: int):
    """Get chat messages for a user and product"""
    try:
        chats = await cache_service.get_chats(user_id, product_id)
        
        return UserChatsResponse(
            user_id=user_id,
            product_id=product_id,
            chats=chats,
            total_messages=len(chats)
        )
        
    except Exception as e:
        logger.error(f"Error in get_user_chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/user/{user_id}/products")
async def get_user_products(user_id: int):
    """Get list of product IDs for a user"""
    try:
        products = await cache_service.get_user_products(user_id)
        return {
            "user_id": user_id,
            "products": products,
            "total_products": len(products)
        }
        
    except Exception as e:
        logger.error(f"Error in get_user_products: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
