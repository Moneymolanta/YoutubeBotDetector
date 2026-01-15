"""
This file contains more advanced implementations of the bot detection models.
These could be integrated into the Flask app for better accuracy.
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import os

class AdvancedBotDetector:
    def __init__(self):
        self.cnn_model = CNNModel()
        self.rnn_model = RNNModel()
        self.bert_model = BERTModel()
        
    def analyze_comment(self, comment_text, like_count=0, reply_count=0):
        # Clean the text
        clean_text = self._clean_text(comment_text)
        
        # Get predictions from each model
        cnn_result = self.cnn_model.predict(clean_text)
        rnn_result = self.rnn_model.predict(clean_text, like_count, reply_count)
        bert_result = self.bert_model.predict(clean_text)
        
        # Determine overall bot likelihood
        bot_votes = sum([
            1 if cnn_result["isBot"] else 0,
            1 if rnn_result["isBot"] else 0,
            1 if bert_result["isBot"] else 0
        ])
        
        is_likely_bot = bot_votes >= 2
        
        return {
            "isLikelyBot": is_likely_bot,
            "botVotes": bot_votes,
            "modelResults": {
                "cnnModel": cnn_result,
                "rnnModel": rnn_result,
                "bertModel": bert_result
            }
        }
    
    def _clean_text(self, text):
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text

class CNNModel:
    def __init__(self):
        # In a real implementation, you would load a trained model here
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # For demonstration, we'll use the rule-based approach
        self.spam_patterns = [
            r'check out my channel',
            r'subscribe to my',
            r'follow me on',
            r'check my profile',
            r'make money',
            r'earn \$\d+',
            r'click here',
            r'bit\.ly',
            r'goo\.gl',
            r't\.co',
            r'cutt\.ly',
            r'tinyurl'
        ]
    
    def predict(self, text):
        # Initialize score (0-100, higher means more likely to be a bot)
        bot_score = 0
        reason = ""
        
        # Check for very short comments
        if len(text) < 5:
            bot_score += 30
            reason = "Very short comment"
        
        # Check for excessive emojis or special characters
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
            u"\U00002700-\U000027BF"  # Dingbats
            "]+", flags=re.UNICODE)
        
        emoji_count = len(emoji_pattern.findall(text))
        text_length = len(text)
        
        if emoji_count > 0 and emoji_count / text_length > 0.3:
            bot_score += 25
            reason = reason + ", Excessive emojis" if reason else "Excessive emojis"
        
        # Check for suspicious links or spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                bot_score += 35
                reason = reason + ", Contains spam patterns" if reason else "Contains spam patterns"
                break
        
        return {
            "isBot": bot_score >= 50,
            "confidence": bot_score,
            "reason": reason if reason else "No specific patterns detected"
        }

class RNNModel:
    def __init__(self):
        # In a real implementation, you would load a trained RNN model here
        self.model = None
    
    def predict(self, text, like_count=0, reply_count=0):
        # Initialize score (0-100, higher means more likely to be a bot)
        bot_score = 0
        reason = ""
        
        # Check comment length (very long comments might be spam)
        if len(text) > 500:
            bot_score += 20
            reason = "Unusually long comment"
        
        # Check engagement metrics
        if like_count == 0 and reply_count == 0 and len(text) > 20:
            bot_score += 25
            reason = reason + ", No engagement" if reason else "No engagement"
        
        # Check for repetitive characters
        if re.search(r'(.)\1{5,}', text):
            bot_score += 30
            reason = reason + ", Repetitive characters" if reason else "Repetitive characters"
        
        # Check for ALL CAPS
        if len(text) > 10 and text == text.upper():
            bot_score += 25
            reason = reason + ", ALL CAPS comment" if reason else "ALL CAPS comment"
        
        return {
            "isBot": bot_score >= 50,
            "confidence": bot_score,
            "reason": reason if reason else "No statistical anomalies detected"
        }

class BERTModel:
    def __init__(self):
        # In a real implementation, you would load a pre-trained BERT model here
        self.model = None
        
        # For demonstration, we'll use a simple approach
        self.bot_phrases = [
            "check this out",
            "you won't believe",
            "click now",
            "free money",
            "make money fast",
            "work from home",
            "thank you for sharing",
            "nice video",
            "great content"
        ]
    
    def predict(self, text):
        # Initialize score (0-100, higher means more likely to be a bot)
        bot_score = 0
        reason = ""
        
        # Check for common bot phrases
        for phrase in self.bot_phrases:
            if phrase in text.lower():
                bot_score += 15
                reason = reason + ", Contains bot phrases" if reason else "Contains bot phrases"
        
        # Check for unnatural language patterns
        words = text.lower().split()
        
        # Check for lack of stopwords (might indicate non-human text)
        stop_words = set(stopwords.words('english'))
        has_stopwords = any(word in stop_words for word in words)
        
        if len(words) > 5 and not has_stopwords:
            bot_score += 20
            reason = reason + ", Unnatural language" if reason else "Unnatural language"
        
        # Check for excessive punctuation
        punctuation_count = len(re.findall(r'[!?.]', text))
        if len(text) > 0 and punctuation_count / len(text) > 0.15:
            bot_score += 20
            reason = reason + ", Excessive punctuation" if reason else "Excessive punctuation"
        
        return {
            "isBot": bot_score >= 40,  # Lower threshold for BERT model
            "confidence": bot_score,
            "reason": reason if reason else "No NLP indicators detected"
        }

# Example usage
if __name__ == "__main__":
    detector = AdvancedBotDetector()
    
    # Test with a likely bot comment
    bot_comment = "Check out my channel for more great content! Subscribe now at bit.ly/12345"
    result = detector.analyze_comment(bot_comment)
    print("Bot Comment Analysis:")
    print(f"Is Likely Bot: {result['isLikelyBot']}")
    print(f"Bot Votes: {result['botVotes']}")
    
    # Test with a likely human comment
    human_comment = "This video was really insightful. I enjoyed the part where you explained the technical details."
    result = detector.analyze_comment(human_comment, like_count=5, reply_count=2)
    print("\nHuman Comment Analysis:")
    print(f"Is Likely Bot: {result['isLikelyBot']}")
    print(f"Bot Votes: {result['botVotes']}")
