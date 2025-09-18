from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
import redis.asyncio as redis
import json

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.on_event("startup")
async def startup():
    # create Redis connection
    r = redis.from_url("redis://localhost:6379", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(r), prefix="fastapi-cache",)

# inner model for "chat"
class Chat(BaseModel):
    message: str
    response: str

# outer model for the request body
class ChatRequest(BaseModel):
    user_id: int
    chat: Chat
    product_id: int

@app.post("/set")
async def manual_cache(payload: ChatRequest):
    print(payload)
    await set_chat(payload.user_id, payload.chat, payload.product_id)

@app.get("/data/{user_id}/{product_id}")
async def get_data(user_id : int, product_id : int) :
    chat = await get_chats(user_id, product_id)
    return chat

async def set_chat(user_id: int, chat, key: int):
    """
    Store chat message for a given user_id and key.
    - Max 5 messages per user_id_key.
    - Max 5 keys per user_id.
    """
    cache = FastAPICache.get_backend()

    redis_key = f"{user_id}_{key}"
    user_keys_key = f"user:{user_id}:keys"

    # --- Manage keys list for user ---
    keys_list = await cache.get(user_keys_key)
    try :
        keys_list = json.loads(keys_list)
    except :
        pass
    if not keys_list:
        keys_list = []

    print(f'type is : {keys_list}' )
    if key not in keys_list:
        keys_list.append(key)
        # Keep only last 5 keys
        if len(keys_list) > 5:
            oldest_key = keys_list.pop(0)
            await cache.clear(key=f"{user_id}_{oldest_key}")

    await cache.set(user_keys_key, json.dumps( keys_list ), expire=1800)

    raw = await cache.get(redis_key)
    chats = json.loads(raw) if raw else []
    if isinstance(chat, BaseModel):
        chat = chat.model_dump()  # or chat.dict() in pydantic v1
    chats.append(chat)
    if len(chats) > 5:
        chats = chats[-5:]

    await cache.set(redis_key, json.dumps(chats, ), expire=1800)

    return key


async def get_chats(user_id : int, key : int):
    cache = FastAPICache.get_backend()
    """Retrieve chats for a given user_id_key."""
    redis_key = f"{user_id}_{key}"
    raw = await cache.get(redis_key)
    return json.loads(raw) if raw else []

async def get_user_keys(user_id : int):
    cache = FastAPICache.get_backend()
    """Retrieve list of keys for a user."""
    user_keys_key = f"user:{user_id}:keys"
    return await cache.get(user_keys_key) or []