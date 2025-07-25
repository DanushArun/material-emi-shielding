"""
Base scraper class with rate limiting, error handling, and common functionality.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
import hashlib
import json
import os
from pathlib import Path


class BaseScraper(ABC):
    """Base class for all scientific literature scrapers."""
    
    def __init__(self, 
                 cache_dir: str = "data_collection/cache",
                 rate_limit_calls: int = 10,
                 rate_limit_period: int = 60,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Initialize base scraper with common functionality.
        
        Args:
            cache_dir: Directory to cache downloaded files
            rate_limit_calls: Number of calls allowed per period
            rate_limit_period: Period in seconds for rate limiting
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Statistics tracking
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'data_extracted': 0,
            'start_time': datetime.now()
        }
        
        # Session for connection pooling
        self.session = None
        
    @abstractmethod
    async def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search for papers/materials related to query.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def extract_data(self, source: Dict) -> Optional[Dict]:
        """
        Extract EMI shielding data from a source.
        Must be implemented by subclasses.
        """
        pass
    
    @sleep_and_retry
    @limits(calls=10, period=60)  # Default rate limit
    async def fetch_url(self, url: str, **kwargs) -> Optional[str]:
        """
        Fetch URL with rate limiting and retries.
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments for the request
            
        Returns:
            Response text or None if failed
        """
        cache_key = self._get_cache_key(url)
        cached_path = self.cache_dir / f"{cache_key}.html"
        
        # Check cache first
        if cached_path.exists():
            self.logger.debug(f"Using cached version of {url}")
            return cached_path.read_text(encoding='utf-8')
        
        self.stats['requests_made'] += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    **kwargs
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Cache successful response
                        cached_path.write_text(text, encoding='utf-8')
                        return text
                    else:
                        self.logger.warning(f"Got status {response.status} for {url}")
                        self.stats['requests_failed'] += 1
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            self.stats['requests_failed'] += 1
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def download_pdf(self, url: str, filename: str) -> Optional[Path]:
        """
        Download PDF file with retries.
        
        Args:
            url: PDF URL
            filename: Local filename to save
            
        Returns:
            Path to downloaded file or None
        """
        pdf_path = self.cache_dir / "pdfs" / filename
        pdf_path.parent.mkdir(exist_ok=True)
        
        if pdf_path.exists():
            self.logger.debug(f"PDF already cached: {filename}")
            return pdf_path
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        content = await response.read()
                        pdf_path.write_bytes(content)
                        self.logger.info(f"Downloaded PDF: {filename}")
                        return pdf_path
                    else:
                        self.logger.warning(f"Failed to download PDF: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error downloading PDF {url}: {e}")
            return None
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def save_extracted_data(self, data: List[Dict], output_file: str):
        """
        Save extracted data to JSON file.
        
        Args:
            data: List of extracted data dictionaries
            output_file: Output filename
        """
        output_path = Path(f"data_collection/output/{output_file}")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(data)} records to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get scraper statistics."""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'runtime_seconds': runtime,
            'requests_made': self.stats['requests_made'],
            'requests_failed': self.stats['requests_failed'],
            'success_rate': (
                (self.stats['requests_made'] - self.stats['requests_failed']) / 
                self.stats['requests_made'] * 100 
                if self.stats['requests_made'] > 0 else 0
            ),
            'data_extracted': self.stats['data_extracted'],
            'extraction_rate': self.stats['data_extracted'] / runtime if runtime > 0 else 0
        }
    
    async def run(self, queries: List[str], max_results_per_query: int = 100):
        """
        Run the scraper with given queries.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
        """
        all_data = []
        
        for query in queries:
            self.logger.info(f"Searching for: {query}")
            
            # Search for sources
            sources = await self.search(query, max_results_per_query)
            self.logger.info(f"Found {len(sources)} sources for '{query}'")
            
            # Extract data from each source
            for source in sources:
                try:
                    data = await self.extract_data(source)
                    if data:
                        data['query'] = query
                        data['source_metadata'] = source
                        all_data.append(data)
                        self.stats['data_extracted'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error extracting data from {source}: {e}")
                    
                # Small delay between extractions
                await asyncio.sleep(1)
        
        # Save all extracted data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.__class__.__name__}_{timestamp}.json"
        self.save_extracted_data(all_data, output_file)
        
        # Print statistics
        stats = self.get_statistics()
        self.logger.info(f"Scraping completed. Statistics: {json.dumps(stats, indent=2)}")
        
        return all_data


class ScraperError(Exception):
    """Custom exception for scraper errors."""
    pass