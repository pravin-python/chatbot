import re
import os
import json
import torch
import warnings
import fasttext
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from langdetect.lang_detect_exception import LangDetectException
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

class AdvancedMultilingualChatbot:
    def __init__(self, intents_file: str = "chatbot/models/intents.json"):
        # Setup cache folder
        self.cache_folder = "./sentence_transformer_cache"
        os.makedirs(self.cache_folder, exist_ok=True)
        
        self.model_names = {
            'primary': 'paraphrase-multilingual-mpnet-base-v2',
            'secondary': 'paraphrase-multilingual-MiniLM-L12-v2',
            'fallback': 'all-MiniLM-L6-v2'
        }
        
        if os.path.exists(intents_file):
            with open(intents_file, "r", encoding="utf-8") as f:
                self.intent_categories = json.load(f)
        else:
            # Fallback to default intents if file not found
            self.intent_categories = self._get_default_intents()
            print(f"⚠️ Using default intents as {intents_file} not found")
        
        self.models = {}
        self.intent_embeddings = {}
        self.language_detector = None
        self.confidence_history = defaultdict(list)
        
    def initialize(self):
        """Initialize the chatbot synchronously"""
        self._load_models()
        self._load_language_detector()
        self._generate_intent_embeddings()
        return self
    
    def _get_default_intents(self) -> Dict:
        """Provide default intents if file is missing"""
        return {
            "product_ingredients": {
                "keywords": ["ingredients", "what's in", "contains", "made of", "composition"],
                "confidence_threshold": 0.6,
                "responses": [
                    "I can help you find the ingredient information for this product.",
                    "आपके प्रोडक्ट की सामग्री के बारे में जानकारी चाहिए?",
                    "Please provide the product details so I can show you the ingredients list."
                ]
            },
            "greeting": {
                "keywords": ["hello", "hi", "hey", "namaste", "hola"],
                "confidence_threshold": 0.5,
                "responses": [
                    "Hello! I'm your product assistant.",
                    "नमस्ते! मैं आपका प्रोडक्ट असिस्टेंट हूँ।",
                    "Hi there! I'm here to help with product information."
                ]
            },
            "unknown": {
                "keywords": [],
                "confidence_threshold": 0.3,
                "responses": [
                    "I'm not sure I understood that correctly.",
                    "मुझे ठीक से समझ नहीं आया।",
                    "I didn't quite catch that."
                ]
            }
        }
    
    def _load_models(self):
        """Load sentence transformer models with error handling"""
        print("🤖 Loading enhanced sentence transformer models...")
        
        for model_key, model_name in self.model_names.items():
            try:
                self.models[model_key] = SentenceTransformer(
                    model_name, 
                    cache_folder=self.cache_folder
                )
                print(f"✅ {model_key} ({model_name}) loaded successfully")
            except Exception as e:
                print(f"❌ Error loading {model_key}: {e}")
                if model_key == 'primary':
                    raise Exception("Primary model failed to load. Cannot continue.")
    
    def _load_language_detector(self):
        """Enhanced language detection with multiple methods"""
        try:
            fasttext_model_path = "./lid.176.bin"
            if not os.path.exists(fasttext_model_path):
                print("📥 FastText model not found. Language detection will be limited to langdetect.")
                self.language_detector = None
                return
                
            self.language_detector = fasttext.load_model(fasttext_model_path)
            print("✅ Language detection model loaded")
        except Exception as e:
            print(f"⚠️ Language detection not available: {e}")
            self.language_detector = None
    
    def _generate_intent_embeddings(self):
        """Generate embeddings for all intent categories and their keywords"""
        print("🧠 Generating intent embeddings...")
        
        for model_key, model in self.models.items():
            self.intent_embeddings[model_key] = {}
            
            for intent_name, intent_data in self.intent_categories.items():
                # Create embeddings for all keywords in this intent
                keywords = intent_data['keywords']
                if keywords:  # Only encode if keywords exist
                    embeddings = model.encode(keywords, convert_to_tensor=True)
                    self.intent_embeddings[model_key][intent_name] = embeddings
        
        print("✅ Intent embeddings generated for all categories")
    
    def detect_language(self, text: str) -> Dict[str, str]:
        """Enhanced language detection with confidence scoring"""
        languages = {}
        
        try:
            lang_detect = detect(text)
            languages['langdetect'] = lang_detect
        except LangDetectException:
            languages['langdetect'] = 'unknown'
        
        if self.language_detector:
            try:
                predictions = self.language_detector.predict(text, k=3)
                fasttext_lang = predictions[0][0].replace('__label__', '')
                confidence = predictions[1][0]
                languages['fasttext'] = f"{fasttext_lang} ({confidence:.3f})"
            except:
                languages['fasttext'] = 'unknown'
        
        languages['script_type'] = self._detect_script_type(text)
        
        return languages
    
    def _detect_script_type(self, text: str) -> str:
        """Detect script type based on character ranges"""
        if re.search(r'[\u0900-\u097F]', text):
            return 'hindi'
        elif re.search(r'[\u0600-\u06FF]', text):
            return 'arabic/urdu'
        elif re.search(r'[\u0A80-\u0AFF]', text):
            return 'gujarati'
        elif re.search(r'[\u0980-\u09FF]', text):
            return 'bengali'
        elif re.search(r'[a-zA-Z]', text):
            return 'english/latin'
        else:
            return 'unknown'
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better matching"""
        text = text.lower().strip()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        replacements = {
            'kya': 'kya', 'hai': 'hai', 'hain': 'hai', 'he': 'hai',
            'mere': 'mera', 'meri': 'mera', 'mujhe': 'main',
            'tumhara': 'tumhara', 'tumhari': 'tumhara',
            'kaisa': 'kaise', 'kaisi': 'kaise', 'kaise': 'kaise'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def find_best_intent(self, user_input: str) -> Dict:
        """Advanced intent detection with ensemble scoring and confidence thresholds"""
        
        processed_input = self._preprocess_text(user_input)
        
        input_embeddings = {}
        for model_key, model in self.models.items():
            try:
                input_embeddings[model_key] = model.encode(processed_input, convert_to_tensor=True)
            except Exception as e:
                print(f"⚠️ Error with {model_key}: {e}")
                continue
        
        intent_scores = {}
        detailed_scores = {}
        
        for intent_name, intent_data in self.intent_categories.items():
            model_similarities = []
            detailed_scores[intent_name] = {}
            
            for model_key, model_embeddings in input_embeddings.items():
                if intent_name in self.intent_embeddings[model_key]:
                    similarities = util.cos_sim(
                        model_embeddings,
                        self.intent_embeddings[model_key][intent_name]
                    )
                    
                    max_similarity = torch.max(similarities).item()
                    model_similarities.append(max_similarity)
                    detailed_scores[intent_name][model_key] = max_similarity
            
            if model_similarities:
                if len(model_similarities) >= 2:
                    ensemble_score = (
                        0.5 * model_similarities[0] +  # Primary model
                        0.3 * model_similarities[1] +  # Secondary model
                        (0.2 * model_similarities[2] if len(model_similarities) > 2 else 0)
                    )
                else:
                    ensemble_score = model_similarities[0]
                
                intent_scores[intent_name] = ensemble_score
        
        if intent_scores:
            best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
            best_score = intent_scores[best_intent]
            confidence_threshold = self.intent_categories[best_intent].get('confidence_threshold', 0.5)
            
            is_confident = best_score >= confidence_threshold
            
            self.confidence_history[best_intent].append(best_score)
            
            return {
                'intent': best_intent,
                'confidence': best_score,
                'threshold': confidence_threshold,
                'is_confident': is_confident,
                'all_scores': intent_scores,
                'detailed_scores': detailed_scores[best_intent],
                'fallback_reason': None if is_confident else f"Confidence {best_score:.3f} below threshold {confidence_threshold}"
            }
        else:
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'threshold': 0.5,
                'is_confident': False,
                'all_scores': {},
                'detailed_scores': {},
                'fallback_reason': 'No intent embeddings available'
            }
    
    def get_contextual_response(self, intent: str, confidence: float, user_input: str) -> str:
        """Generate contextual responses based on detected intent"""
        
        # Use responses from intents file if available
        if intent in self.intent_categories and 'responses' in self.intent_categories[intent]:
            response_options = self.intent_categories[intent]['responses']
        else:
            # Fallback responses
            responses = {
                'product_ingredients': [
                    "I can help you find the ingredient information for this product. Could you share the product name or barcode?",
                    "आपके प्रोडक्ट की सामग्री के बारे में जानकारी चाहिए? कृपया प्रोडक्ट का नाम या बारकोड बताएं।",
                    "Please provide the product details so I can show you the ingredients list."
                ],
                'unknown': [
                    "I'm not sure I understood that correctly. Could you rephrase your question about the product?",
                    "मुझे ठीक से समझ नहीं आया। क्या आप अपना सवाल दूसरे तरीके से पूछ सकते हैं?",
                    "I didn't quite catch that. Could you ask your question differently?"
                ]
            }
            response_options = responses.get(intent, responses['unknown'])
        
        # Choose response based on confidence level
        if confidence > 0.8:
            return response_options[0]  # Most confident English response
        elif confidence > 0.6 and len(response_options) > 1:
            return response_options[1]  # Hindi response if available
        else:
            return response_options[-1]  # Fallback response
    
    def analyze_conversation(self, user_input: str) -> Dict:
        """Complete conversation analysis with enhanced features"""
        
        # Language detection
        detected_languages = self.detect_language(user_input)
        
        # Intent detection
        intent_result = self.find_best_intent(user_input)
        
        # Generate response
        response = self.get_contextual_response(
            intent_result['intent'],
            intent_result['confidence'],
            user_input
        )
        
        # Additional analysis
        analysis = {
            'input': user_input,
            'processed_input': self._preprocess_text(user_input),
            'languages': detected_languages,
            'intent_analysis': intent_result,
            'response': response,
            'recommendations': self._get_recommendations(intent_result),
            'timestamp': self._get_timestamp()
        }
        
        return analysis
    
    def _get_recommendations(self, intent_result: Dict) -> List[str]:
        """Provide recommendations based on intent analysis"""
        recommendations = []
        
        if not intent_result['is_confident']:
            recommendations.append("Try rephrasing your question for better understanding")
            recommendations.append("Use specific product names or keywords")
        
        if intent_result['confidence'] < 0.4:
            recommendations.append("Your question might be outside my knowledge area")
            recommendations.append("Try asking about product ingredients, nutrition, or barcode information")
        
        intent = intent_result['intent']
        if intent in ['product_ingredients', 'product_nutrition', 'barcode_information']:
            recommendations.append("Provide product name or barcode for more accurate information")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_detailed_analysis(self, analysis: Dict):
        """Enhanced result display with better formatting"""
        print(f"\n{'='*80}")
        print(f"🎯 CHATBOT ANALYSIS - {analysis['timestamp']}")
        print(f"{'='*80}")
        
        print(f"📝 Original Input: '{analysis['input']}'")
        print(f"🔧 Processed Input: '{analysis['processed_input']}'")
        
        print(f"\n🌐 Language Detection:")
        for method, result in analysis['languages'].items():
            print(f"   {method.title()}: {result}")
        
        intent_data = analysis['intent_analysis']
        print(f"\n🎯 Intent Detection:")
        print(f"   📌 Detected Intent: {intent_data['intent'].upper()}")
        print(f"   📊 Confidence: {intent_data['confidence']:.4f}")
        print(f"   🎚️ Threshold: {intent_data['threshold']}")
        print(f"   ✅ Is Confident: {'YES' if intent_data['is_confident'] else 'NO'}")
        
        if intent_data['fallback_reason']:
            print(f"   ⚠️ Fallback Reason: {intent_data['fallback_reason']}")
        
        if intent_data['detailed_scores']:
            print(f"\n📈 Model Scores for '{intent_data['intent']}':")
            for model, score in intent_data['detailed_scores'].items():
                print(f"   {model.title()}: {score:.4f}")
        
        if intent_data['all_scores']:
            sorted_intents = sorted(
                intent_data['all_scores'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            print(f"\n🏆 Top Intent Candidates:")
            for i, (intent, score) in enumerate(sorted_intents, 1):
                print(f"   {i}. {intent}: {score:.4f}")
        
        print(f"\n🤖 Bot Response:")
        print(f"   {analysis['response']}")
        
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print(f"{'='*80}\n")