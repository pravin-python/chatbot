from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Chat(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    response: str = Field(..., min_length=1, max_length=1000)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    message: str = Field(..., min_length=1, max_length=1000)
    product_id: int = Field(..., gt=0)

class ChatResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Chat]] = None

class UserChatsResponse(BaseModel):
    user_id: int
    product_id: int
    chats: List[Chat]
    total_messages: int
