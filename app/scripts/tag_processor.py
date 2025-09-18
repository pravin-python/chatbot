from multilingual.chatbot.async_utils import AsyncChatbotWrapper

async def main(message: str):
    try:
        chatbot = await AsyncChatbotWrapper(intents_file="intents.json").initialize()
        analysis = await chatbot.analyze_conversation_async(message)
        result = analysis['intent_analysis']
        if result['is_confident']:
            if result['intent'] == 'product_ingredients':
                return "Action: Redirecting to ingredients database... product_ingredients"
            elif result['intent'] == 'product_nutrition':
                return "Action: Continuing conversation... product_nutrition"
            elif result['intent'] == 'product_harmful_effects':
                return "Action: Continuing conversation... product_harmful_effects"
            elif result['intent'] == 'product_suitability':
                return "Action: Continuing conversation... product_suitability"
            elif result['intent'] == 'barcode_information':
                return "Action: Continuing conversation... barcode_information"
            elif result['intent'] == 'personal_name':
                return "Action: Continuing conversation... personal_name"
            elif result['intent'] == 'personal_age':
                return "Action: Continuing conversation.. personal_age."
            elif result['intent'] == 'personal_height':
                return "Action: Continuing conversation...personal_height"
            elif result['intent'] == 'greeting':
                return "Action: Continuing conversation... greeting"
            elif result['intent'] == 'how_are_you':
                return "Action: Continuing conversation... how_are_you"
            elif result['intent'] == 'daily_wellness':
                return "Action: Continuing conversation... daily_wellness"
            elif result['intent'] == 'product_alternatives':
                return "Action: Continuing conversation... product_alternatives"
            else:
                return "⚠️ Action: Intent not recognized. Please try again."
        else:
            return "⚠️ Action: Intent not recognized. Please try again."
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await chatbot.close()