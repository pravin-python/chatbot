import json
import logging
from typing import List, Optional, Union
from datetime import datetime
from app.core.redis_manager import redis_manager
from app.config import settings
from app.services.response_generator import response_generator

logger = logging.getLogger(__name__)

class CacheService:
    
    async def process_and_store_chat(self, user_id: int, message: str, product_id: int) -> dict:
        """
        Main function: Process message and store with generated response
        """
        try:
            # cache = redis_manager.get_backend()
            # redis_key = f"{user_id}_{product_id}"
            # raw = await cache.get(redis_key)
            # print(f"Raw data from Redis for key {redis_key}: {raw}")
            
            existing_chats = await self.get_chats(user_id, product_id)
            response = await response_generator.generate_response(
                message=message,
                user_id=user_id,
                product_id=product_id,
                chat_history=existing_chats
            )
            
            # 3. Create chat object
            chat_data = {
                "message": message,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
            # 4. Store chat with all logic
            success = await self.set_chat(user_id, chat_data, product_id)
            
            if success:
                return {
                    "success": True,
                    "chat": chat_data,
                    "total_messages": len(existing_chats) + 1
                }
            else:
                return {"success": False, "error": "Failed to store chat"}
                
        except Exception as e:
            logger.error(f"Error in process_and_store_chat: {e}")
            return {"success": False, "error": str(e)}
    
    async def set_chat(self, user_id: int, chat: Union[dict], product_id: int) -> bool:
        """Store chat with max 5 messages limit"""
        try:
            cache = redis_manager.get_backend()
            redis_key = f"{user_id}_{product_id}"
            user_keys_key = f"user:{user_id}:keys"
            
            await self._manage_user_keys(cache, user_keys_key, product_id, user_id)
            
            raw = await cache.get(redis_key)
            chats = json.loads(raw) if raw else []
            
            chats.append(chat)
            
            if len(chats) > settings.max_messages_per_chat:
                chats = chats[-settings.max_messages_per_chat:]
                logger.info(f"Removed old messages, keeping last {settings.max_messages_per_chat}")
            
            await cache.set(
                redis_key, 
                json.dumps(chats, default=str), 
                expire=settings.cache_expire_time
            )
            
            logger.info(f"Chat stored successfully for user {user_id}, product {product_id}. Total messages: {len(chats)}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chat: {e}")
            return False
    
    async def get_chats(self, user_id: int, product_id: int) -> List[dict]:
        """Retrieve chats for a given user_id and product_id"""
        try:
            cache = redis_manager.get_backend()
            redis_key = f"{user_id}_{product_id}"
            
            raw = await cache.get(redis_key)
            chats = json.loads(raw) if raw else []
            
            logger.info(f"Retrieved {len(chats)} chats for user {user_id}, product {product_id}")
            return chats
            
        except Exception as e:
            logger.error(f"Error retrieving chats: {e}")
            return []
    
    async def get_user_products(self, user_id: int) -> List[int]:
        """Retrieve list of product IDs for a user"""
        try:
            cache = redis_manager.get_backend()
            user_keys_key = f"user:{user_id}:keys"
            
            raw = await cache.get(user_keys_key)
            if raw:
                try:
                    return json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return []
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving user products: {e}")
            return []
    
    async def _manage_user_keys(self, cache, user_keys_key: str, product_id: int, user_id: int):
        """Manage the list of product keys for a user (max 5 products)"""
        keys_list = await cache.get(user_keys_key)
        
        if keys_list:
            try:
                keys_list = json.loads(keys_list)
            except (json.JSONDecodeError, TypeError):
                keys_list = []
        else:
            keys_list = []
        
        if product_id not in keys_list:
            keys_list.append(product_id)
            
            # Keep only last 5 products (FIFO)
            if len(keys_list) > settings.max_products_per_user:
                oldest_key = keys_list.pop(0)  # Remove first (oldest)
                await cache.clear(key=f"{user_id}_{oldest_key}")
                logger.info(f"Removed oldest product {oldest_key} for user {user_id}")
        
        await cache.set(
            user_keys_key, 
            json.dumps(keys_list), 
            expire=settings.cache_expire_time
        )

cache_service = CacheService()
