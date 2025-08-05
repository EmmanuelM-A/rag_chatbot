"""
Web search module for retrieving information from the internet when no
relevant documents are found.
"""
import os

import requests
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

from config import (
    MAX_WEB_SEARCH_RESULTS,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from src.components.ingestion.process_documents import FileDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger

logger = get_logger("web_search_logger")

SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")


class WebSearcher:
    """
    Handles web search functionality and content retrieval.
    """

    def __init__(self):
        self.search_api_key = SEARCH_API_KEY
        self.search_engine_id = SEARCH_ENGINE_ID
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def search_web(self, query: str,
                   num_results: int = MAX_WEB_SEARCH_RESULTS) -> List[
        Dict[str, Any]]:
        """
        Perform web search using Google Custom Search API.

        Args:
            query: Search query
            num_results: Number of results to retrieve

        Returns:
            List of search results with title, snippet, and url
        """

        if not self.search_api_key or not self.search_engine_id:
            logger.error("Search API credentials not configured")
            return self._fallback_search(query, num_results)

        logger.info(f"Performing web search for: {query}")

        try:
            # Google Custom Search API endpoint
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.search_api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10)  # API limit is 10 per request
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'items' not in data:
                logger.warning("No search results found")
                return []

            results = []
            for item in data['items']:
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'url': item.get('link', ''),
                    'source': 'google_search'
                })

            logger.info(f"Retrieved {len(results)} search results")
            return results

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return self._fallback_search(query, num_results)

    def _fallback_search(self, query: str, num_results: int) -> List[
        Dict[str, Any]]:
        """
        Fallback search using DuckDuckGo (no API key required).

        Note: This is a simple implementation. For production, consider using
        dedicated libraries like `duckduckgo-search` or similar.
        """
        logger.info("Using fallback search method")

        try:
            # Simple DuckDuckGo search (note: this may not work reliably in production)
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            results = []

            # Parse DuckDuckGo results (simplified)
            result_elements = soup.find_all('div', class_='result')[
                              :num_results]

            for element in result_elements:
                title_elem = element.find('a', class_='result__a')
                snippet_elem = element.find('div', class_='result__snippet')

                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'snippet': snippet_elem.get_text(strip=True),
                        'url': title_elem.get('href', ''),
                        'source': 'duckduckgo_search'
                    })

            logger.info(f"Retrieved {len(results)} fallback search results")
            return results

        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []

    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract text content from a web page.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract text from main content areas
            content_selectors = [
                'main', 'article', '.content', '.post-content',
                '.entry-content', '.article-content', 'p'
            ]

            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = " ".join(
                        [elem.get_text(strip=True) for elem in elements])
                    break

            if not content_text:
                # Fallback to body text
                body = soup.find('body')
                if body:
                    content_text = body.get_text(strip=True)

            # Clean up the text
            content_text = " ".join(content_text.split())

            if len(content_text) > 100:  # Minimum viable content length
                logger.debug(
                    f"Successfully extracted {len(content_text)} characters from {url}")
                return content_text
            else:
                logger.warning(f"Insufficient content extracted from {url}")
                return None

        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return None

    def search_and_retrieve_content(self, query: str) -> List[FileDocument]:
        """
        Perform web search and retrieve full content from result URLs.

        Args:
            query: Search query

        Returns:
            List of FileDocument objects with web content
        """
        logger.info(f"Searching web and retrieving content for: {query}")

        # Get search results
        search_results = self.search_web(query)

        if not search_results:
            logger.warning("No search results to process")
            return []

        documents = []

        for i, result in enumerate(search_results):
            try:
                # Add a small delay to be respectful to servers
                if i > 0:
                    time.sleep(1)

                url = result.get('url', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')

                if not url:
                    continue

                # Fetch full page content
                content = self.fetch_page_content(url)

                if content:
                    # Combine title, snippet, and content
                    full_content = f"Title: {title}\n\nSummary: {snippet}\n\nContent: {content}"

                    # Create document with metadata
                    metadata = {
                        'source': url,
                        'title': title,
                        'type': 'web_search',
                        'query': query,
                        'search_source': result.get('source', 'unknown')
                    }

                    document = FileDocument(full_content, metadata)
                    documents.append(document)

                    logger.debug(f"Retrieved content from {url}")
                else:
                    # If we can't get full content, use just the snippet
                    snippet_content = f"Title: {title}\n\nSummary: {snippet}"
                    metadata = {
                        'source': url,
                        'title': title,
                        'type': 'web_search_snippet',
                        'query': query,
                        'search_source': result.get('source', 'unknown')
                    }

                    document = FileDocument(snippet_content, metadata)
                    documents.append(document)

                    logger.debug(f"Using snippet for {url}")

            except Exception as e:
                logger.error(f"Error processing search result {i}: {e}")
                continue

        logger.info(f"Retrieved {len(documents)} documents from web search")
        return documents

    def chunk_web_documents(self, documents: List[FileDocument]) -> List[
        FileDocument]:
        """
        Chunk web documents for embedding.

        Args:
            documents: List of FileDocument objects from web search

        Returns:
            List of chunked FileDocument objects
        """
        logger.debug("Chunking web documents")

        chunks = []

        for doc in documents:
            try:
                doc_chunks = self.text_splitter.split_text(doc.content)

                for chunk_text in doc_chunks:
                    if chunk_text.strip():  # Skip empty chunks
                        # Create new metadata for each chunk
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata['chunk_type'] = 'web_content'

                        chunk_doc = FileDocument(chunk_text, chunk_metadata)
                        chunks.append(chunk_doc)

            except Exception as e:
                logger.error(f"Error chunking web document: {e}")
                continue

        logger.info(f"Created {len(chunks)} chunks from web documents")
        return chunks
