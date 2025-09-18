import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

class NLPProcessor:
    def __init__(self):
        self.nlp_models = {}
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._setup_nlp()
    
    def _setup_nlp(self):
        """Setup NLP tools with error handling"""
        try:
            # Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize spaCy
            try:
                self.nlp_models['en'] = spacy.load("en_core_web_sm")
            except:
                print("⚠️ English spaCy model not found. Some NLP features will be limited.")
            
            # Initialize sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Initialize NER
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            print("✅ NLP processor initialized successfully")
            
        except Exception as e:
            print(f"⚠️ NLP setup failed: {e}")
    
    async def analyze_sentiment_async(self, text: str) -> Dict[str, Any]:
        """Async sentiment analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze_sentiment, text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the input"""
        try:
            if not self.sentiment_analyzer:
                return {'label': 'NEUTRAL', 'score': 0.5}
                
            result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    async def extract_entities_async(self, text: str) -> List[Dict]:
        """Async entity extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.extract_entities, text)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        try:
            if not self.ner_pipeline:
                return []
                
            entities = self.ner_pipeline(text[:512])  # Limit text length
            return [
                {
                    'entity': entity['entity_group'],
                    'word': entity['word'],
                    'score': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                }
                for entity in entities
            ]
        except Exception as e:
            print(f"⚠️ Entity extraction error: {e}")
            return []
    
    def extract_product_info(self, text: str) -> Dict[str, List]:
        """Extract product-related information"""
        entities = self.extract_entities(text)
        product_info = {
            'product_names': [],
            'brands': [],
            'quantities': [],
            'ingredients_mentioned': []
        }
        
        for entity in entities:
            if entity['entity'] in ['PRODUCT', 'MISC']:
                product_info['product_names'].append(entity['word'])
            elif entity['entity'] == 'ORG':
                product_info['brands'].append(entity['word'])
            elif entity['entity'] == 'QUANTITY':
                product_info['quantities'].append(entity['word'])
        
        # Check for common ingredient mentions
        ingredient_keywords = ['ingredient', 'component', 'material', 'content', 'contains']
        for keyword in ingredient_keywords:
            if keyword in text.lower():
                product_info['ingredients_mentioned'].append(keyword)
        
        return product_info
    
    def enhanced_preprocess_text(self, text: str) -> str:
        """More sophisticated text preprocessing"""
        # Basic preprocessing
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenization and lemmatization
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        except:
            pass  # Continue without stopword removal if not available
        
        return ' '.join(tokens)
    
    def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown()