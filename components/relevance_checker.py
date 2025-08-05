"""
Determine if queries are relevant to the document corpus.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from components.prompt_loader import create_prompt_template
from config import (
    DOCUMENT_TOPICS_FILE_PATH,
    TOPIC_RELEVANCE_THRESHOLD,
    LLM_TEMPERATURE,
    TOPIC_ANALYSIS_PROMPT_FILEPATH
)
from utils.logger import get_logger

class RelevanceChecker:
    """
    Handles relevance checking for queries against the document corpus.
    """

    def __init__(self, embedding_model_name, llm_model_name):
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.llm = ChatOpenAI(model=llm_model_name, temperature=LLM_TEMPERATURE)
        self.document_topics = self._load_document_topics()
        self.logger = get_logger(__name__)

    def _load_document_topics(self) -> Dict[str, Any]:
        """Load document topics from disk if they exist."""

        if not os.path.exists(DOCUMENT_TOPICS_FILE_PATH):
            return {}

        try:
            with open(DOCUMENT_TOPICS_FILE_PATH, "rb") as f:
                topics = pickle.load(f)

            self.logger.debug("Document topics loaded from disk.")

            return topics
        except ValueError as e:
            self.logger.error(f"Error loading document topics: {e}")
            return {}

    def save_document_topics(self, topics: Dict[str, Any]) -> None:
        """Save document topics to disk."""

        try:
            os.makedirs(os.path.dirname(DOCUMENT_TOPICS_FILE_PATH), exist_ok=True)

            with open(DOCUMENT_TOPICS_FILE_PATH, "wb") as f:
                pickle.dump(topics, f)

            self.logger.debug("Document topics saved to disk.")
        except ValueError as e:
            self.logger.error(f"Error saving document topics: {e}")

    def extract_document_topics(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Extract main topics and themes from the document corpus.

        Args:
            documents: List of FileDocument objects

        Returns:
            Dictionary containing topics, keywords, and embeddings
        """

        self.logger.debug("Extracting topics from document corpus...")

        # Combine all document content for topic extraction
        combined_content = "\n\n".join(
            [doc.content[:500] for doc in documents[:10]]
        )  # Sample first 500 chars of first 10 docs

        topic_prompt = create_prompt_template(TOPIC_ANALYSIS_PROMPT_FILEPATH)

        # Extract topics using LLM
        chain = topic_prompt | self.llm | StrOutputParser()
        topic_analysis = chain.invoke({"content": combined_content})

        # Parse the response
        topics_data = self._parse_topic_analysis(topic_analysis)

        # Create embeddings for topics and keywords
        all_terms = topics_data.get("main_topics", []) + topics_data.get(
            "keywords", [])
        if all_terms:
            term_embeddings = self.embedding_model.embed_documents(all_terms)
            topics_data["embeddings"] = {
                term: embedding for term, embedding in
                zip(all_terms, term_embeddings)
            }

        # Create a general domain embedding
        domain_text = (f"{topics_data.get('domain', '')} "
                       f"{' '.join(topics_data.get('main_topics', []))}")

        if domain_text.strip():
            domain_embedding = self.embedding_model.embed_query(domain_text)
            topics_data["domain_embedding"] = domain_embedding

        self.document_topics = topics_data
        self.save_document_topics(topics_data)

        self.logger.info(f"Extracted topics: {topics_data.get('main_topics', [])}")

        return topics_data

    def _parse_topic_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse the LLM's topic analysis response."""

        topics_data = {
            "main_topics": [],
            "keywords": [],
            "domain": ""
        }

        lines = analysis.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("MAIN_TOPICS:"):
                current_section = "main_topics"
                content = line.replace("MAIN_TOPICS:", "").strip()
                if content:
                    topics_data["main_topics"] = [t.strip() for t in
                                                  content.strip("[]").split(
                                                      ",")]
            elif line.startswith("KEYWORDS:"):
                current_section = "keywords"
                content = line.replace("KEYWORDS:", "").strip()
                if content:
                    topics_data["keywords"] = [k.strip() for k in
                                               content.strip("[]").split(",")]
            elif line.startswith("DOMAIN:"):
                current_section = "domain"
                content = line.replace("DOMAIN:", "").strip()
                topics_data["domain"] = content
            elif current_section and line:
                # Handle multi-line content
                if current_section == "main_topics" and line not in \
                        topics_data["main_topics"]:
                    topics_data["main_topics"].extend(
                        [t.strip() for t in line.strip("[]").split(",")])
                elif current_section == "keywords" and line not in topics_data[
                    "keywords"]:
                    topics_data["keywords"].extend(
                        [k.strip() for k in line.strip("[]").split(",")])

        # Clean up empty strings
        topics_data["main_topics"] = [t for t in topics_data["main_topics"] if
                                      t]
        topics_data["keywords"] = [k for k in topics_data["keywords"] if k]

        return topics_data

    def is_query_relevant(
            self,
            query: str,
            similarity_threshold: float = TOPIC_RELEVANCE_THRESHOLD
    ) -> bool:
        """
        Determine if a query is relevant to the document corpus.

        Args:
            query: User's query string.
            similarity_threshold: Minimum similarity score for relevance.

        Returns:
            True if query is relevant, False otherwise.
        """

        if not self.document_topics:
            self.logger.warning(
                "No document topics available. Assuming query is relevant.")
            return True

        self.logger.debug(f"Checking relevance for query: {query}")

        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Check similarity with domain embedding
        if "domain_embedding" in self.document_topics:
            domain_similarity = self._cosine_similarity(
                query_embedding,
                self.document_topics["domain_embedding"]
            )

            if domain_similarity >= similarity_threshold:
                self.logger.info(
                    f"Query relevant to domain (similarity: {domain_similarity:.3f})")
                return True

        # Check similarity with individual topics and keywords
        if "embeddings" in self.document_topics:
            max_similarity = 0
            best_match = ""

            for term, term_embedding in self.document_topics[
                "embeddings"].items():
                similarity = self._cosine_similarity(query_embedding,
                                                     term_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = term

            if max_similarity >= similarity_threshold:
                self.logger.info(
                    f"Query relevant to '{best_match}' (similarity: {max_similarity:.3f})")
                return True

        # Use LLM as final check for semantic relevance
        semantic_relevance = self._check_semantic_relevance(query)
        if semantic_relevance:
            self.logger.info("Query deemed relevant through semantic analysis")
            return True

        self.logger.info(
            f"Query not relevant to document corpus (max similarity: {max_similarity:.3f})")
        return False

    def _cosine_similarity(
            self,
            vec1: List[float],
            vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _check_semantic_relevance(self, query: str) -> bool:
        """Use LLM to perform semantic relevance checking."""
        if not self.document_topics:
            return False

        topics_str = ", ".join(self.document_topics.get("main_topics", []))
        domain = self.document_topics.get("domain", "")
        keywords_str = ", ".join(
            self.document_topics.get("keywords", [])[:10])

        if not topics_str and not domain and not keywords_str:
            self.logger.warning("No topics available for semantic relevance check")
            return True

        relevance_prompt = create_prompt_template(
            "../prompts/semantic_relevance_checking_prompt.yaml"
        )

        try:
            chain = relevance_prompt | self.llm | StrOutputParser()
            response = chain.invoke({})

            is_relevant = response.strip().upper().startswith("RELEVANT")
            self.logger.debug(f"LLM relevance check: {response}")
            return is_relevant

        except Exception as e:
            self.logger.error(f"Error in semantic relevance check: {e}")
            return False
