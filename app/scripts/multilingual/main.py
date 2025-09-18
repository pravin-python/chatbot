#!/usr/bin/env python3
"""
Main entry point for the Advanced Multilingual Chatbot
"""

import asyncio
import sys
import os

# Add the chatbot package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chatbot'))

from chatbot.async_utils import AsyncChatbotWrapper

async def main():
    print("🚀 Initializing Advanced Multilingual Chatbot...")
    print("="*80)
    
    try:
        # Initialize the async chatbot
        chatbot = await AsyncChatbotWrapper(
            intents_file="chatbot/models/intents.json"
        ).initialize()
        
        print("\n✅ Chatbot setup completed successfully!")
        print(f"\n📋 Available Intent Categories: {list(chatbot.chatbot.intent_categories.keys())}")
        
        # Test examples
        test_queries = [
            "hello, what ingredients are in this product?",
            "मेरे प्रोडक्ट में क्या है?",
            "is this product safe for diabetics?",
            "what's the barcode for this item?"
        ]
        
        print(f"\n🧪 Testing with sample queries...")
        for query in test_queries:
            print(f"\nTesting: '{query}'")
            analysis = await chatbot.analyze_conversation_async(query)
            chatbot.chatbot.print_detailed_analysis(analysis)
        
        # Interactive mode
        print("\n💬 Interactive Mode - Enter your queries (type 'quit' to exit):")
        while True:
            print("\n" + "-"*50)
            try:
                query = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Thanks for using the chatbot! Goodbye!")
                break
            
            if query.lower() in ['quit', 'exit', 'q', 'bye']:
                print("👋 Thanks for using the chatbot! Goodbye!")
                break
            
            if not query:
                print("⚠️ Please enter a valid query.")
                continue
            
            try:
                analysis = await chatbot.analyze_conversation_async(query)
                print(analysis)
                chatbot.chatbot.print_detailed_analysis(analysis)
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await chatbot.close()

if __name__ == "__main__":
    asyncio.run(main())