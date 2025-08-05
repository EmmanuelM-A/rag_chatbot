"""
Responsible for loading ChatPromptTemplates from prompt files.
"""

import yaml
from langchain.prompts import ChatPromptTemplate
from utils.logger import get_logger

logger = get_logger(__name__)


def create_prompt_template(prompt_filepath):
    """
    This function expects the YAML file to define two keys: 'system' and 'user'.
    - 'system': Instructions for the assistant's behavior (e.g., tone, constraints)
    - 'user': Template for the user query (should contain placeholders like {context}, {query})

    :param:
        prompt_filepath (str): Path to the YAML file containing the prompt messages.

    :return:
        ChatPromptTemplate: A LangChain-compatible prompt template that can be used with LLM chains.

    Example YAML format:
        system: |
            You are a helpful assistant...
        user: |
            Context: {context}
            Question: {query}
    """

    # Open and parse the YAML prompt file
    with open(prompt_filepath, "r") as f:
        prompt_data = yaml.safe_load(f)

    logger.debug("Prompt loaded and prompt template created.")

    # Create a LangChain ChatPromptTemplate using the parsed message templates
    return ChatPromptTemplate.from_messages([
        ("system", prompt_data["system"]),  # Set assistant behavior
        ("user", prompt_data["user"])       # Set user query format
    ])
