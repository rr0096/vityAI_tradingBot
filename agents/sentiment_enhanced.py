# agents/sentiment_enhanced.py
from __future__ import annotations

from models.outputs import SentimentOutput # Assuming this is correctly defined elsewhere
import logging
import re
from typing import ClassVar, Dict, List, Any, Literal, Optional, Tuple, Deque as TypingDeque
from datetime import datetime, timedelta, timezone
from collections import Counter, deque
import statistics
import hashlib
import numpy as np
import json # For the schema in prompt
from scipy import stats
from config.config_loader import APP_CONFIG
from utils.agent_memory import get_last_agent_decision

from crewai import Agent
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from tools.news_scraper import fetch_news
from tools.twitter_scraper import TwitterScraper
from tools.reddit_scraper import RedditScraper
from tools.fear_greed import FearGreedTool
from tools.alternative_news import AlternativeNewsProvider

from .enhanced_base_llm_agent import EnhancedBaseLLMAgent

logger = logging.getLogger(__name__)

# Initialize tools
_twitter_scraper = TwitterScraper()
_reddit_scraper = RedditScraper()
_fear_greed_tool = FearGreedTool()
_alternative_news_provider = AlternativeNewsProvider()

class TextItem(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, arbitrary_types_allowed=True)
    content: str
    source: Literal["news", "twitter", "reddit"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_timestamp: Optional[datetime] = None
    word_count: int = 0
    quality_score: float = Field(default=0.0)  # Range 0.0 to 1.0
    content_hash: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.content:
            self.content = ""
            self.word_count = 0
            self.quality_score = 0.0
            self.content_hash = hashlib.md5("".encode('utf-8')).hexdigest()[:16]
        else:
            self.word_count = len(self.content.split())
            if self.original_timestamp and self.original_timestamp.tzinfo is None:
                self.original_timestamp = self.original_timestamp.replace(tzinfo=timezone.utc)
            if not self.timestamp.tzinfo:
                self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
            
            self.quality_score = self._calculate_quality()
            self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()[:16]

    def _calculate_quality(self) -> float:
        if not self.content or len(self.content.strip()) < 10: return 0.0
        score = 0.0
        content_lower = self.content.lower()

        # Word count contribution
        if 20 <= self.word_count <= 150: score += 0.30
        elif 10 <= self.word_count <= 250: score += 0.15
        else: score += 0.05

        # Keyword relevance
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'trading', 'market',
            'price', 'bull', 'bear', 'sec', 'etf', 'rally', 'dump', 'ath', 'fud', 'fomo',
            'binance', 'coinbase', 'solana', 'ripple', 'xrp', 'cardano', 'ada', 'doge',
            'halving', 'defi', 'nft', 'altcoin', 'whale', 'regulation', 'fed', 'inflation'
        ]
        keyword_matches = sum(1 for keyword in crypto_keywords if keyword in content_lower)
        score += min(keyword_matches * 0.07, 0.40) # Max 0.4 from keywords

        # Spam indicators (penalty)
        spam_indicators = ['🚀', '💎', '1000x', 'to the moon', 'must buy', 'giveaway', 'free money', 'guaranteed profit']
        excessive_caps = sum(1 for char in self.content if char.isupper()) > (len(self.content) * 0.4) and len(self.content) > 30
        if any(indicator in content_lower for indicator in spam_indicators) or "!!!" in self.content or excessive_caps:
            score *= 0.5 # Penalize heavily for spam

        # Source bonus
        if self.source == "news": score = min(1.0, score + 0.30) # Slightly higher bonus for news
        elif self.source == "reddit": score = min(1.0, score + 0.05)
        # Twitter might be neutral or slightly penalized if quality is often low

        # Recency
        effective_timestamp = self.original_timestamp or self.timestamp
        if effective_timestamp.tzinfo is None: # Ensure tz-aware
            effective_timestamp = effective_timestamp.replace(tzinfo=timezone.utc)
        
        hours_old = (datetime.now(timezone.utc) - effective_timestamp).total_seconds() / 3600
        if hours_old < 3: score = min(1.0, score + 0.15)
        elif hours_old < 12: score = min(1.0, score + 0.05)
        elif hours_old > 72: score *= 0.6 # Older than 3 days
        elif hours_old > 48: score *= 0.7 # Older than 2 days
        elif hours_old > 24: score *= 0.85 # Older than 1 day
        
        return round(min(max(0.0, score), 1.0), 3)

class LLMSentimentResponse(BaseModel):
    # This Pydantic model defines the expected JSON structure from the LLM
    overall_sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    positive_texts_count: int = Field(default=0)  # Must be >= 0
    negative_texts_count: int = Field(default=0)  # Must be >= 0
    neutral_texts_count: int = Field(default=0)   # Must be >= 0
    reasoning: str = Field(..., min_length=10) # Ensure reasoning is provided

class EnhancedSentimentAnalyst(EnhancedBaseLLMAgent):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="ignore", validate_assignment=False)

    name: ClassVar[str] = "EnhancedSentimentAnalyst"
    role: ClassVar[str] = "Advanced Market Sentiment Analyzer for Crypto News and Social Media"
    goal: ClassVar[str] = (
        "Analyze diverse textual data (news, Twitter, Reddit) to provide a high-quality, "
        "contextualized sentiment analysis (POSITIVE, NEGATIVE, NEUTRAL) with confidence scores, "
        "and identify key trends and influencing factors like Fear & Greed index."
    )
    backstory: ClassVar[str] = (
        "An AI-powered analyst specializing in dissecting the nuanced language of financial markets, "
        "particularly within the volatile crypto space. It leverages advanced NLP techniques to sift "
        "through noise, identify genuine sentiment signals, and quantify market mood, adapting its "
        "analysis to various data sources and their inherent biases."
    )
    
    # These will be set as instance attributes in __init__ to avoid Pydantic conflicts

    def __init__(self, **data: Any):
        data.setdefault('agent_type', 'sentiment')
        super().__init__(**data)
        
        # Set configuration as instance attributes to avoid Pydantic Field conflicts
        self.texts_to_sample_for_llm: int = 15  # Reduced to manage token limits
        self.max_history_age_days_for_trend: int = 1  # For short-term trend
        
        # Cache configuration for sentiment data (slower changing)
        self.cache_refresh_interval_minutes: int = 10  # Refresh every 10 minutes
        self._last_refresh_time: Optional[datetime] = None
        self._is_data_cached: bool = False
        
        # Initialize internal state after super().__init__
        self._all_fetched_text_items: List[TextItem] = []
        self._sampled_texts_for_llm: List[TextItem] = []
        self._current_fear_greed_value: int = 50
        self._last_full_analysis_output: Optional[SentimentOutput] = None
        self._analysis_history: List[Tuple[datetime, SentimentOutput]] = []
        self._processed_content_hashes: TypingDeque[str] = deque(maxlen=2000)
        self.max_texts_per_source: int = 30  # Changed from _max_texts_to_fetch_per_source
        self.min_text_quality_threshold: float = 0.25  # Changed from _min_text_quality_threshold
        
        self.refresh() # Initial data fetch

    def _fetch_and_process_source(
        self,
        fetch_func: callable,
        source_name: Literal["news", "twitter", "reddit"],
        content_key: str = "title" # Default for news, 'text' for twitter/reddit
    ) -> List[TextItem]:
        items: List[TextItem] = []
        try:
            raw_data = fetch_func() # Call the scraper
            
            data_list_for_processing: List[Any] = []
            if isinstance(raw_data, list): # News, or direct list from scraper
                data_list_for_processing = raw_data
            elif isinstance(raw_data, dict) and source_name in ["twitter", "reddit"]: # Twitter/Reddit often return dicts
                temp_list = []
                for user_or_sub_items in raw_data.values():
                    if isinstance(user_or_sub_items, list):
                        temp_list.extend(user_or_sub_items)
                data_list_for_processing = temp_list
            else:
                logger.warning(f"[{self.agent_type}] Unexpected data structure from {source_name}: {type(raw_data)}. Raw: {str(raw_data)[:100]}")
                return items

            fetched_count = 0
            for item_data in data_list_for_processing:
                if fetched_count >= self.max_texts_per_source:
                    break
                
                content_str = ""
                original_ts_raw = None

                if isinstance(item_data, dict):
                    # Try common keys for content
                    content_str = str(item_data.get('content', item_data.get('text', item_data.get(content_key, ""))))
                    original_ts_raw = item_data.get('timestamp', item_data.get('created_utc', item_data.get('published_at')))
                elif isinstance(item_data, str): # Simple list of strings
                    content_str = item_data
                
                if content_str and content_str.strip():
                    original_ts = datetime.now(timezone.utc) # Default to now
                    if original_ts_raw:
                        try:
                            if isinstance(original_ts_raw, (int, float)): # Unix timestamp
                                original_ts = datetime.fromtimestamp(original_ts_raw, tz=timezone.utc)
                            elif isinstance(original_ts_raw, str):
                                ts_str = str(original_ts_raw).replace('Z', '+00:00')
                                original_ts = datetime.fromisoformat(ts_str)
                            # Ensure timezone if still naive
                            if original_ts.tzinfo is None:
                                original_ts = original_ts.replace(tzinfo=timezone.utc)
                        except (ValueError, TypeError) as e_ts:
                            logger.debug(f"Could not parse timestamp '{original_ts_raw}' for {source_name}: {e_ts}. Defaulting to now.")
                    
                    items.append(TextItem(content=content_str, source=source_name, original_timestamp=original_ts))
                    fetched_count += 1
            
            logger.info(f"[{self.agent_type}] Fetched {len(items)} items from {source_name}.")
        except Exception as e:
            logger.warning(f"[{self.agent_type}] Scraper for {source_name} failed: {e}", exc_info=True)
        return items

    def _process_alternative_news(self, alternative_news: List[Dict[str, Any]]) -> List[TextItem]:
        """Process alternative news data into TextItem objects."""
        items: List[TextItem] = []
        
        try:
            for news_item in alternative_news:
                if not isinstance(news_item, dict):
                    continue
                    
                # Extract content from title and optionally description
                title = news_item.get('title', '')
                description = news_item.get('description', '')
                content = f"{title}. {description}".strip()
                
                if not content:
                    continue
                
                # Parse timestamp
                original_ts = datetime.now(timezone.utc)  # Default
                published_at = news_item.get('published_at')
                if published_at:
                    try:
                        if isinstance(published_at, str):
                            # Handle various date formats
                            published_at = published_at.replace('Z', '+00:00')
                            original_ts = datetime.fromisoformat(published_at)
                            if original_ts.tzinfo is None:
                                original_ts = original_ts.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not parse alternative news timestamp '{published_at}': {e}")
                
                items.append(TextItem(
                    content=content,
                    source="news",
                    original_timestamp=original_ts
                ))
            
            logger.info(f"[{self.agent_type}] Processed {len(items)} alternative news items.")
        except Exception as e:
            logger.warning(f"[{self.agent_type}] Failed to process alternative news: {e}", exc_info=True)
        
        return items

    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status information."""
        now = datetime.now(timezone.utc)
        
        if self._last_refresh_time is None:
            return {
                "is_cached": False,
                "last_refresh": "never",
                "time_since_refresh": "never",
                "next_refresh_in": f"{self.cache_refresh_interval_minutes} minutes",
                "cache_interval_minutes": self.cache_refresh_interval_minutes,
                "needs_refresh": True
            }
        
        time_since_refresh = now - self._last_refresh_time
        time_until_next = timedelta(minutes=self.cache_refresh_interval_minutes) - time_since_refresh
        
        return {
            "is_cached": self._is_data_cached,
            "last_refresh": self._last_refresh_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
            "time_since_refresh": f"{int(time_since_refresh.total_seconds() / 60)} minutes",
            "next_refresh_in": f"{max(0, int(time_until_next.total_seconds() / 60))} minutes",
            "cache_interval_minutes": self.cache_refresh_interval_minutes,
            "needs_refresh": self._should_refresh_cache()
        }

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed based on time interval."""
        if self._last_refresh_time is None:
            return True  # First time, needs refresh
        
        time_since_last_refresh = datetime.now(timezone.utc) - self._last_refresh_time
        return time_since_last_refresh.total_seconds() >= (self.cache_refresh_interval_minutes * 60)
    
    def _mark_cache_refreshed(self) -> None:
        """Mark cache as refreshed with current timestamp."""
        self._last_refresh_time = datetime.now(timezone.utc)
        self._is_data_cached = True

    def refresh(self, force: bool = False) -> None:
        """
        Refresh sentiment data sources with intelligent caching.
        
        Args:
            force: If True, bypass cache and force refresh regardless of time interval
        """
        # Check if we need to refresh based on cache interval
        if not force and not self._should_refresh_cache():
            logger.info(
                f"[{self.name}] Cache still valid "
                f"(last refresh: {self._last_refresh_time.strftime('%H:%M:%S') if self._last_refresh_time else 'never'}, "
                f"interval: {self.cache_refresh_interval_minutes}min). Skipping refresh."
            )
            return
        
        logger.info(f"[{self.name}] Refreshing data sources...")
        all_new_texts: List[TextItem] = []
        
        # Fetch news using simplified news scraper (which uses alternative sources)
        logger.info(f"[{self.name}] Fetching news...")
        all_new_texts.extend(self._fetch_and_process_source(fetch_news, "news", content_key="title"))
        
        # Fetch Twitter (Nitter)
        # Nitter can be unreliable; handle potential empty results gracefully
        twitter_content_key = 'text' # Assuming Nitter scraper returns 'text'
        all_new_texts.extend(self._fetch_and_process_source(_twitter_scraper._run, "twitter", content_key=twitter_content_key))
        
        # Fetch Reddit
        reddit_content_key = 'text' # Assuming Reddit scraper returns 'text' or similar
        all_new_texts.extend(self._fetch_and_process_source(_reddit_scraper._run, "reddit", content_key=reddit_content_key))

        self._all_fetched_text_items = all_new_texts # Store all, even if not high quality, for stats
        
        filtered_texts = self._deduplicate_and_filter_texts(all_new_texts)
        self._sampled_texts_for_llm = self._smart_sample_texts(filtered_texts)
        
        try:
            fg_value_str = _fear_greed_tool._run()
            self._current_fear_greed_value = int(fg_value_str) if fg_value_str and fg_value_str.isdigit() else 50
            logger.info(f"[{self.name}] Fear & Greed Index updated: {self._current_fear_greed_value}")
        except Exception as e:
            logger.warning(f"[{self.name}] FearGreed API failed: {e}. Using default value 50.", exc_info=True)
            self._current_fear_greed_value = 50
            
        logger.info(
            f"[{self.name}] Refresh complete. Fetched: {len(self._all_fetched_text_items)} "
            f"-> Filtered: {len(filtered_texts)} -> Sampled for LLM: {len(self._sampled_texts_for_llm)}"
        )
        
        # Mark cache as refreshed
        self._mark_cache_refreshed()
        logger.info(f"[{self.name}] Cache updated. Next refresh in {self.cache_refresh_interval_minutes} minutes.")

    def _deduplicate_and_filter_texts(self, fetched_texts: List[TextItem]) -> List[TextItem]:
        unique_texts_this_batch: Dict[str, TextItem] = {} # Use dict for easier deduplication by hash
        
        # Sort by quality and recency to prioritize better texts if hashes collide (though unlikely for short hashes)
        def get_sort_key(text_item: TextItem):
            ts = text_item.original_timestamp or text_item.timestamp
            if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
            return (text_item.quality_score, ts) # Higher quality, then more recent
        
        sorted_texts = sorted(fetched_texts, key=get_sort_key, reverse=True)

        final_filtered_list: List[TextItem] = []
        for text_item in sorted_texts:
            if text_item.quality_score < self.min_text_quality_threshold:
                continue
            if text_item.content_hash not in self._processed_content_hashes and \
               text_item.content_hash not in unique_texts_this_batch:
                unique_texts_this_batch[text_item.content_hash] = text_item
                final_filtered_list.append(text_item)
        
        # Update global processed hashes (deque handles maxlen automatically)
        for h in unique_texts_this_batch.keys():
            self._processed_content_hashes.append(h)
        
        logger.info(
            f"[{self.agent_type}] Deduplication & Filtering: {len(fetched_texts)} initial "
            f"-> {len(final_filtered_list)} unique & high-quality texts for this batch. "
            f"Total processed hashes in memory: {len(self._processed_content_hashes)}"
        )
        return final_filtered_list
        
    def _smart_sample_texts(self, texts: List[TextItem]) -> List[TextItem]:
        if not texts: return []
        if len(texts) <= self.texts_to_sample_for_llm: return texts
        
        now_utc = datetime.now(timezone.utc)
        
        def composite_score(text_item: TextItem) -> float:
            quality = text_item.quality_score # Already 0-1
            
            effective_timestamp = text_item.original_timestamp or text_item.timestamp
            if effective_timestamp.tzinfo is None:
                effective_timestamp = effective_timestamp.replace(tzinfo=timezone.utc)
            
            hours_old = (now_utc - effective_timestamp).total_seconds() / 3600
            # Recency score: 1 for fresh, decaying to 0 over ~72 hours
            recency_score = max(0, 1 - (hours_old / 72.0))
            
            source_bonus = {"news": 0.2, "reddit": 0.05, "twitter": 0.0}.get(text_item.source, 0.0)
            
            # Weighted score: Quality (60%), Recency (30%), Source (10%)
            return (quality * 0.6) + (recency_score * 0.3) + (source_bonus * 0.1)

        # Sort by the composite score, descending
        texts.sort(key=composite_score, reverse=True)
        
        return texts[:self.texts_to_sample_for_llm]

    def _extract_top_keywords(self, texts: List[TextItem], top_k: int = 7) -> List[str]:
        if not texts: return []
        
        all_text_content = " ".join([item.content.lower() for item in texts])
        
        # Expanded base keywords, more crypto-specific
        base_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency', 'blockchain',
            'bullish', 'bearish', 'market', 'price', 'trading', 'exchange', 'binance',
            'coinbase', 'sec', 'etf', 'halving', 'defi', 'nft', 'altcoin', 'solana', 'sol',
            'ripple', 'xrp', 'cardano', 'ada', 'dogecoin', 'doge', 'shiba', 'shib',
            'rally', 'crash', 'dump', 'pump', 'ath', 'dip', 'correction', 'support',
            'resistance', 'fud', 'fomo', 'hodl', 'whale', 'arbitrage', 'regulation',
            'federal reserve', 'fed', 'inflation', 'interest rate', 'mining', 'staking',
            'buy', 'sell', 'hold', 'long', 'short', 'investment', 'analysis', 'prediction', 'outlook',
            'launch', 'airdrop', 'scam', 'hack', 'security', 'wallet', 'custody'
        }
        
        # Basic stopword list (can be expanded or use a library like NLTK for more robust stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'to', 'and', 'of', 'in', 'it', 'for', 'on', 'with',
            'this', 'that', 'be', 'will', 'was', 'are', 'as', 'at', 'by', 'from', 'has',
            'have', 'he', 'she', 'they', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'it\'s', 'its', 'about', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
            'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        words = re.findall(r'\b[a-z]{3,15}\b', all_text_content) # Basic word tokenization
        
        word_counts = Counter()
        for w in words:
            if w not in stopwords:
                if w in base_keywords:
                    word_counts[w] += 2 # Give more weight to predefined crypto keywords
                else:
                    word_counts[w] += 1
                    
        return [word for word, _ in word_counts.most_common(top_k)]

    def run(self, *args, **kwargs):
        last_decision, last_ts = get_last_agent_decision('sentiment', APP_CONFIG.trading.symbol)
        if last_decision and last_ts:
            logger.info(f"[EnhancedSentimentAnalyst] Última decisión reciente: {last_decision} a las {last_ts}")
        
        """
        Analyzes sentiment from fetched texts using an LLM.
        Returns a SentimentOutput object.
        """
        # Ensure data is fresh if needed, or rely on external refresh calls
        # self.refresh() # Optional: uncomment if each run should always fetch newest data

        fg_value = self._current_fear_greed_value
        # Fear/Greed influence: 0 for neutral (50), 1 for extreme (0 or 100)
        fg_influence_factor = round(abs(fg_value - 50) / 50.0, 3)
        
        avg_quality = statistics.mean([t.quality_score for t in self._sampled_texts_for_llm]) if self._sampled_texts_for_llm else 0.0
        total_initial_texts = len(self._all_fetched_text_items)
        top_keywords = self._extract_top_keywords(self._sampled_texts_for_llm)
        current_trend = self.get_sentiment_trend_label(hours_back=24)

        if not self._sampled_texts_for_llm:
            logger.warning(f"[{self.name}] No texts available for LLM analysis after sampling.")
            # Return a default neutral SentimentOutput
            return SentimentOutput(
                overall_sentiment="NEUTRAL",
                positive_texts_count=0, negative_texts_count=0, neutral_texts_count=0,
                reasoning="No texts sampled for LLM analysis. Source data might be unavailable or filtered out.",
                confidence_score=0.1, # Low confidence due to no data
                fear_greed_value_used=fg_value,
                fear_greed_influence_factor=fg_influence_factor,
                avg_data_quality_score=round(avg_quality, 3),
                total_texts_analyzed_by_llm=0,
                total_texts_fetched_initially=total_initial_texts,
                top_keywords_found=top_keywords,
                sentiment_trend_short_term=current_trend
            )
        
        # Prepare text samples for the prompt
        text_samples_for_prompt_list = []
        now_utc = datetime.now(timezone.utc)
        for t in self._sampled_texts_for_llm:
            ts_for_hours = t.original_timestamp or t.timestamp
            if ts_for_hours.tzinfo is None: ts_for_hours = ts_for_hours.replace(tzinfo=timezone.utc)
            hours_ago = (now_utc - ts_for_hours).total_seconds() / 3600
            # Truncate individual text content to manage prompt length
            truncated_content = t.content[:200] + '...' if len(t.content) > 200 else t.content
            text_samples_for_prompt_list.append(
                f"- SRC: {t.source.upper()} | Q: {t.quality_score:.2f} | AGE: {hours_ago:.1f}h | TXT: {truncated_content}"
            )
        text_samples_for_prompt_str = "\n".join(text_samples_for_prompt_list)

        # Define the Pydantic model for the LLM's expected response structure
        # This is already defined as LLMSentimentResponse class variable

        # Construct the prompt with clear instructions for JSON output
        prompt = f"""Eres un analista experto en sentimiento del mercado de criptomonedas.
Tu tarea es analizar los siguientes textos recopilados de noticias y redes sociales.
Determina el sentimiento general (POSITIVE, NEGATIVE, o NEUTRAL), cuenta cuántos textos caen en cada categoría,
y proporciona un breve razonamiento para tu análisis general.

Textos para analizar (Fuente | Calidad | Antigüedad | Contenido):
{text_samples_for_prompt_str}

Información adicional de contexto:
- Índice Fear & Greed actual: {fg_value} (0-100, >50 es Greed, <50 es Fear)
- Palabras clave destacadas en los textos: {', '.join(top_keywords) if top_keywords else "N/A"}
- Tendencia de sentimiento reciente (últimas 24h): {current_trend}

CRITICAL INSTRUCTIONS FOR RESPONSE FORMATTING:
1. Tu respuesta DEBE SER ÚNICAMENTE un objeto JSON válido.
2. No incluyas ningún texto, explicación, disculpa o formato markdown (como ```json ... ```) fuera del objeto JSON.
3. Todo el resultado debe ser el objeto JSON, adhiriéndose estrictamente al siguiente esquema Pydantic:
{json.dumps(LLMSentimentResponse.model_json_schema(), indent=2)}

Ejemplo del formato JSON exacto requerido:
{{
  "overall_sentiment": "NEUTRAL",
  "positive_texts_count": 2,
  "negative_texts_count": 1,
  "neutral_texts_count": 2,
  "reasoning": "Análisis conciso basado en los textos, considerando el F&G y palabras clave."
}}

Ahora, proporciona ÚNICAMENTE el objeto JSON basado en tu análisis:"""
        
        # Query the LLM using the enhanced validation method
        # This integrates constitutional prompting and JSON validation
        llm_response_obj = self._query_llm_with_validation(
            prompt,
            LLMSentimentResponse,
            schema_type="sentiment",  # Use sentiment schema instead of trading
            use_constitutional=True,
            temperature=0.1 # Lower temperature for more deterministic JSON output
        )
        
        if not llm_response_obj or not isinstance(llm_response_obj, LLMSentimentResponse):
            logger.error(f"[{self.name}] Failed to get a valid structured response from LLM. Response: {llm_response_obj}")
            error_reason = getattr(llm_response_obj, 'reasoning', "Error obtaining or parsing LLM response for sentiment.") if llm_response_obj else "LLM response was None or invalid type."
            
            # Try to get counts if they exist, even if main object failed validation
            pos_c = getattr(llm_response_obj, 'positive_texts_count', 0) if llm_response_obj else 0
            neg_c = getattr(llm_response_obj, 'negative_texts_count', 0) if llm_response_obj else 0
            neu_c = getattr(llm_response_obj, 'neutral_texts_count', 0) if llm_response_obj else 0
            
            return SentimentOutput(
                overall_sentiment="NEUTRAL", # Default on error
                reasoning=error_reason,
                confidence_score=0.1, # Low confidence on error
                positive_texts_count=pos_c,
                negative_texts_count=neg_c,
                neutral_texts_count=neu_c,
                fear_greed_value_used=fg_value,
                fear_greed_influence_factor=fg_influence_factor,
                avg_data_quality_score=round(avg_quality, 3),
                total_texts_analyzed_by_llm=(pos_c + neg_c + neu_c), # Sum of what might have been parsed
                total_texts_fetched_initially=total_initial_texts,
                top_keywords_found=top_keywords,
                sentiment_trend_short_term=current_trend
            )
        
        # Successfully got a structured response
        pos_count = llm_response_obj.positive_texts_count
        neg_count = llm_response_obj.negative_texts_count
        neu_count = llm_response_obj.neutral_texts_count
        total_analyzed_by_llm = pos_count + neg_count + neu_count
        
        # Calculate confidence score
        # Clarity factor: 1 if no neutral texts, 0 if all neutral (among those analyzed by LLM)
        clarity_denominator = total_analyzed_by_llm if total_analyzed_by_llm > 0 else 1
        clarity_factor = 1.0 - (neu_count / clarity_denominator) if clarity_denominator > 0 else 0.0
        
        # Confidence: avg data quality (60%), clarity of LLM's classification (30%), F&G non-extremity (10%)
        confidence = (avg_quality * 0.6) + (clarity_factor * 0.3) + ((1 - fg_influence_factor) * 0.1)
        if llm_response_obj.overall_sentiment == "NEUTRAL": # Slightly reduce confidence for neutral calls
            confidence *= 0.85

        final_output = SentimentOutput(
            overall_sentiment=llm_response_obj.overall_sentiment,
            positive_texts_count=pos_count,
            negative_texts_count=neg_count,
            neutral_texts_count=neu_count,
            reasoning=llm_response_obj.reasoning,
            confidence_score=round(min(max(0.1, confidence), 0.95), 3), # Clamp confidence
            fear_greed_value_used=fg_value,
            fear_greed_influence_factor=fg_influence_factor,
            avg_data_quality_score=round(avg_quality, 3),
            total_texts_analyzed_by_llm=total_analyzed_by_llm,
            total_texts_fetched_initially=total_initial_texts,
            top_keywords_found=top_keywords,
            sentiment_trend_short_term=current_trend
        )
        
        self._last_full_analysis_output = final_output
        self._analysis_history.append((datetime.now(timezone.utc), final_output))
        # Deque handles maxlen automatically for _analysis_history if it's a deque
            
        logger.info(
            f"[{self.name}] Sentiment Analysis successful. Overall: {final_output.overall_sentiment}, "
            f"Conf: {final_output.confidence_score:.2f}, Texts (P/Ng/Nt): {pos_count}/{neg_count}/{neu_count}"
        )
        return final_output

    def get_sentiment_trend_label(self, hours_back: int = 24) -> Literal["IMPROVING", "DETERIORATING", "STABLE", "INSUFFICIENT_DATA"]:
        if len(self._analysis_history) < 2:
            return "INSUFFICIENT_DATA"
        
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=hours_back)
        
        # Filter recent analyses and ensure timestamps are timezone-aware
        recent_analyses_valid_ts: List[Tuple[datetime, SentimentOutput]] = []
        for ts, out_obj in self._analysis_history:
            if ts.tzinfo is None: # Should not happen if stored correctly
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff_time:
                recent_analyses_valid_ts.append((ts, out_obj))
        
        if len(recent_analyses_valid_ts) < 2:
            return "INSUFFICIENT_DATA"
            
        # Convert sentiment to numerical values (-1, 0, 1) weighted by confidence
        sentiment_values: List[float] = []
        for _, out_obj in recent_analyses_valid_ts:
            val = 0
            if out_obj.overall_sentiment == "POSITIVE": val = 1
            elif out_obj.overall_sentiment == "NEGATIVE": val = -1
            sentiment_values.append(val * out_obj.confidence_score)
            # Using confidence_score as a weight for the trend calculation

        if len(sentiment_values) < 2: # Need at least two points for a trend
            return "INSUFFICIENT_DATA"
            
        try:
            # Simple linear regression to find the trend slope
            x_axis = np.arange(len(sentiment_values))
            if len(set(sentiment_values)) == 1: # All values are the same, no trend
                slope = 0.0
            else:
                slope, _, _, _, _ = stats.linregress(x_axis, sentiment_values)
            
            # Define thresholds for trend labels
            if slope > 0.10: return "IMPROVING"
            elif slope < -0.10: return "DETERIORATING"
            return "STABLE"
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating sentiment trend with numpy/scipy: {e}", exc_info=True)
            return "STABLE" # Default to stable on error

    def get_last_full_analysis(self) -> Optional[SentimentOutput]:
        return self._last_full_analysis_output

