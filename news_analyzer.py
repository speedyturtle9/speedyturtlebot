# utils/news_feed_analyzer.py

import logging
import xmltodict
import requests
import json
from datetime import datetime, timedelta
from textblob import TextBlob
from trading_bot.utils.config import NEWS_FEEDS, NEWS_KEYWORDS, NEWS_CACHE_DURATION, NEWS_SENTIMENT_THRESHOLD, NEWS_IMPACT_THRESHOLD

class NewsFeedAnalyzer:
    def __init__(self):
        """Initialize News Feed Analyzer"""
        self.news_cache = {}
        self.last_update = {}
        self.feeds = NEWS_FEEDS
        self.keywords = NEWS_KEYWORDS
        self.cache_file = "database/news_cache/news_cache.json"
        self.load_news_cache(self.cache_file)
        logging.info("Initialized News Feed Analyzer")

    def get_rss_feed(self, feed_url):
        """Fetch RSS feed content"""
        try:
            response = requests.get(feed_url, timeout=10)
            return xmltodict.parse(response.content)
        except Exception as e:
            logging.error(f"Error fetching RSS feed {feed_url}: {str(e)}")
            return None

    def save_news_cache(self, filepath):
        """Save news cache to file"""
        try:
            with open(filepath, 'w') as file:
                cache_data = {
                    'cache': self.news_cache,
                    'last_update': {k: v.isoformat() for k, v in self.last_update.items()}
                }
                json.dump(cache_data, file, indent=4)
            logging.info(f"News cache saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving news cache: {str(e)}")

    def load_news_cache(self, filepath):
        """Load news cache from file"""
        try:
            with open(filepath, 'r') as file:
                cache_data = json.load(file)
                self.news_cache = cache_data['cache']
                self.last_update = {k: datetime.fromisoformat(v) 
                                  for k, v in cache_data['last_update'].items()}
            logging.info(f"News cache loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading news cache: {str(e)}")
            self.news_cache = {}
            self.last_update = {}

    def analyze_news_for_instrument(self, instrument):
        """Analyze news relevant to a specific trading instrument"""
        try:
            # Update news cache if needed
            self.update_news_cache(instrument)

            relevant_news = []
            sentiment_score = 0
            news_impact = 0

            # Get relevant keywords for the instrument
            keywords = self.keywords.get(instrument, [])

            # Analyze cached news
            for news_item in self.news_cache.get(instrument, []):
                if self.is_news_relevant(news_item, keywords):
                    # Analyze sentiment
                    sentiment = self.analyze_sentiment(news_item)
                    impact = self.calculate_news_impact(news_item)

                    relevant_news.append({
                        'title': news_item['title'],
                        'description': news_item['description'],
                        'link': news_item['link'],
                        'sentiment': sentiment,
                        'impact': impact,
                        'timestamp': news_item.get('pubDate', '')
                    })

                    sentiment_score += sentiment
                    news_impact += impact

            # Calculate average sentiment and impact
            if relevant_news:
                avg_sentiment = sentiment_score / len(relevant_news)
                avg_impact = news_impact / len(relevant_news)
            else:
                avg_sentiment = 0
                avg_impact = 0

            # Save updated cache
            self.save_news_cache(self.cache_file)

            return {
                'sentiment': avg_sentiment,
                'impact': avg_impact,
                'news_items': relevant_news,
                'should_trade': abs(avg_sentiment) <= NEWS_SENTIMENT_THRESHOLD and 
                               avg_impact <= NEWS_IMPACT_THRESHOLD
            }

        except Exception as e:
            logging.error(f"Error analyzing news for {instrument}: {str(e)}")
            return None

    def update_news_cache(self, instrument):
        """Update news cache for an instrument"""
        try:
            current_time = datetime.now()
            last_update = self.last_update.get(instrument)

            if last_update is None or (current_time - last_update) > timedelta(minutes=NEWS_CACHE_DURATION):
                news_items = []

                # Fetch news from all relevant feeds
                for feed_url in self.feeds.get(instrument, []):
                    feed_data = self.get_rss_feed(feed_url)
                    if feed_data and 'rss' in feed_data:
                        items = feed_data['rss']['channel'].get('item', [])
                        news_items.extend(items)

                self.news_cache[instrument] = news_items
                self.last_update[instrument] = current_time
                logging.info(f"Updated news cache for {instrument}")

        except Exception as e:
            logging.error(f"Error updating news cache: {str(e)}")

    def analyze_sentiment(self, news_item):
        """Analyze sentiment of news item"""
        try:
            text = f"{news_item['title']} {news_item['description']}"
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {str(e)}")
            return 0

    def calculate_news_impact(self, news_item):
        """Calculate potential market impact of news"""
        try:
            # Define impact keywords and their weights
            impact_keywords = {
                'high': ['crisis', 'crash', 'emergency', 'war', 'disaster', 'recession'],
                'medium': ['change', 'update', 'announcement', 'report', 'policy'],
                'low': ['minor', 'small', 'routine', 'regular', 'normal']
            }

            text = f"{news_item['title']} {news_item['description']}".lower()
            impact_score = 0

            # Calculate impact based on keyword presence
            for word in text.split():
                if word in impact_keywords['high']:
                    impact_score += 1.0
                elif word in impact_keywords['medium']:
                    impact_score += 0.5
                elif word in impact_keywords['low']:
                    impact_score += 0.1

            return min(impact_score, 1.0)  # Normalize to 0-1 range

        except Exception as e:
            logging.error(f"Error calculating news impact: {str(e)}")
            return 0

    def is_news_relevant(self, news_item, keywords):
        """Check if news item is relevant based on keywords"""
        try:
            text = f"{news_item['title']} {news_item['description']}".lower()
            return any(keyword.lower() in text for keyword in keywords)
        except Exception as e:
            logging.error(f"Error checking news relevance: {str(e)}")
            return False

    def get_recent_news_summary(self, instrument):
        """Get a summary of recent news for an instrument"""
        try:
            news_items = self.news_cache.get(instrument, [])[:5]  # Get 5 most recent news items
            summary = f"Recent News for {instrument}:\n\n"
            
            for item in news_items:
                summary += f"Title: {item['title']}\n"
                summary += f"Description: {item['description']}\n"
                summary += f"Link: {item['link']}\n"
                summary += "-" * 50 + "\n"
                
            return summary
            
        except Exception as e:
            logging.error(f"Error getting news summary: {str(e)}")
            return "Error getting news summary"