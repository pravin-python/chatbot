import asyncio
import aiohttp
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from .core import AdvancedMultilingualChatbot
from .nlp_processor import NLPProcessor

class AsyncChatbotWrapper:
    def __init__(self, intents_file: str = "chatbot/models/intents.json"):
        self.chatbot = AdvancedMultilingualChatbot(intents_file)
        self.nlp_processor = NLPProcessor()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.session = None
    
    async def initialize(self):
        """Async initialization"""
        # Initialize chatbot components in parallel
        chatbot_init = asyncio.get_event_loop().run_in_executor(
            self.executor, self.chatbot.initialize
        )
        
        # We can add other async initialization tasks here if needed
        
        await chatbot_init
        print("✅ Async chatbot initialized successfully")
        return self
    
    async def analyze_conversation_async(self, user_input: str) -> Dict[str, Any]:
        """Async conversation analysis with enhanced NLP features"""
        print(f"\n🔍 Analyzing input: {user_input}")
        # Run language detection and intent analysis in parallel
        language_task = asyncio.get_event_loop().run_in_executor(
            self.executor, self.chatbot.detect_language, user_input
        )
        
        intent_task = asyncio.get_event_loop().run_in_executor(
            self.executor, self.chatbot.find_best_intent, user_input
        )
        
        # Run NLP analysis in parallel
        sentiment_task = self.nlp_processor.analyze_sentiment_async(user_input)
        entities_task = self.nlp_processor.extract_entities_async(user_input)
        
        # Wait for all tasks to complete
        detected_languages, intent_result, sentiment, entities = await asyncio.gather(
            language_task, intent_task, sentiment_task, entities_task
        )
        
        # Extract product info (fast operation, can be synchronous)
        product_info = self.nlp_processor.extract_product_info(user_input)
        
        # Generate response
        response = self.chatbot.get_contextual_response(
            intent_result['intent'],
            intent_result['confidence'],
            user_input
        )
        
        # Compile analysis
        analysis = {
            'input': user_input,
            'processed_input': self.chatbot._preprocess_text(user_input),
            'languages': detected_languages,
            'intent_analysis': intent_result,
            'sentiment': sentiment,
            'entities': entities,
            'product_info': product_info,
            'response': response,
            'recommendations': self.chatbot._get_recommendations(intent_result),
            'timestamp': self.chatbot._get_timestamp()
        }
        
        return analysis
    
    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.executor.shutdown()
        self.nlp_processor.shutdown()