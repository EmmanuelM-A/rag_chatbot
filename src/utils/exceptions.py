"""
Custom exceptions for the RAG chatbot application. Provides comprehensive
error handling for all components.
"""
import json
from typing import Optional, Any, Dict, List
import traceback
from pathlib import Path


class RAGChatbotError(Exception):
    """
    Base exception for RAG chatbot errors.
    All custom exceptions inherit from this base class.
    """

    def __init__(
        self,
        message: str = "An error occurred during the RAG Chatbot Pipeline.",
        error_code: str = None,
        context: Dict[str, Any] = None,
        original_error: Exception = None
    ) -> None:
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.original_error = original_error
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None
        }

    def to_json(self) -> str:
        """
        Serialize the error as a JSON string.
        """

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)

    def as_log(self):
        """
        Format the error for logging output.
        """

        log_msg = f"[{self.error_code}] {self.message}"

        if self.context:
            log_msg += f" | Context: {self.context}"

        if self.original_error:
            log_msg += f" | Original: {self.original_error}"

        return log_msg


# ============================================================================
# CONFIGURATION & INITIALIZATION ERRORS
# ============================================================================

class SettingsConfigError(RAGChatbotError, ValueError):
    """
    Raised when an error occurs during setting/config initialization.
    """

    def __init__(
        self,
        setting_name: str,
        issue: str,
        suggestion: str = None
    ) -> None:
        """
        Initializes the SettingConfigError instance.

        Args:
            setting_name: The name of setting that caused the error.
            issue: The issue you wish to address.
            suggestion: Any possible suggestions to resolve the issue.
        """

        self.setting_name = setting_name
        self.issue = issue
        self.suggestion = suggestion

        message = f"Configuration error for '{setting_name}': {issue}"
        if suggestion:
            message += f" Suggestion: {suggestion}"

        super().__init__(
            message=message,
            error_code="SETTINGS_CONFIG_ERROR",
            context={
                "setting_name": setting_name,
                "issue": issue,
                "suggestion": suggestion
            }
        )


class EnvironmentVariableError(SettingsConfigError):
    """Raised when required environment variables are missing or invalid."""
    def __init__(self, var_name: str, reason: str = "missing"):
        super().__init__(
            setting_name=var_name,
            issue=f"Environment variable is {reason}",
            suggestion=f"Set the {var_name} environment variable"
        )


class DependencyError(RAGChatbotError):
    """Raised when required dependencies are missing or incompatible."""
    def __init__(self, dependency: str, issue: str, version_required: str = None):
        message = f"Dependency error: {dependency} - {issue}"
        if version_required:
            message += f" (requires version {version_required})"

        super().__init__(
            message=message,
            error_code="DEPENDENCY_ERROR",
            context={
                "dependency": dependency,
                "issue": issue,
                "version_required": version_required
            }
        )


# ============================================================================
# FILE & DIRECTORY ERRORS
# ============================================================================

class FileSystemError(RAGChatbotError, OSError):
    """Base class for file system related errors."""
    def __init__(self, path: str, operation: str, reason: str):
        self.path = path
        self.operation = operation
        self.reason = reason

        super().__init__(
            message=f"File system error: Cannot {operation} '{path}' - {reason}",
            error_code="FILESYSTEM_ERROR",
            context={
                "path": path,
                "operation": operation,
                "reason": reason
            }
        )


class DirectoryNotFoundError(FileSystemError):
    """Raised when a required directory does not exist."""
    def __init__(self, path: str):
        super().__init__(
            path=path,
            operation="access directory",
            reason="directory does not exist"
        )


class InvalidDirectoryError(FileSystemError):
    """Raised when the path provided is not a directory."""
    def __init__(self, path: str):
        super().__init__(
            path=path,
            operation="access directory",
            reason="path is not a directory"
        )


class FileDoesNotExist(FileSystemError):
    """Raised when a file cannot be found or does not exist."""
    def __init__(self, path: str, context_info: str = None):
        reason = "file does not exist"
        if context_info:
            reason += f" ({context_info})"

        super().__init__(
            path=path,
            operation="access file",
            reason=reason
        )


class FilePermissionError(FileSystemError):
    """Raised when there are insufficient permissions to access a file."""
    def __init__(self, path: str, operation: str):
        super().__init__(
            path=path,
            operation=operation,
            reason="insufficient permissions"
        )


class FileCorruptedError(FileSystemError):
    """Raised when a file is corrupted or has invalid format."""
    def __init__(self, path: str, file_type: str, details: str = None):
        reason = f"corrupted or invalid {file_type} format"
        if details:
            reason += f" - {details}"

        super().__init__(
            path=path,
            operation="read file",
            reason=reason
        )


class FileTypeNotSupported(RAGChatbotError):
    """Raised when a file type is not supported."""
    def __init__(self, file_type: str, supported_types: List[str] = None):
        message = f"File type '{file_type}' is not supported"
        if supported_types:
            message += f". Supported types: {', '.join(supported_types)}"

        super().__init__(
            message=message,
            error_code="UNSUPPORTED_FILE_TYPE",
            context={
                "file_type": file_type,
                "supported_types": supported_types
            }
        )


# ============================================================================
# DOCUMENT PROCESSING ERRORS
# ============================================================================

class DocumentProcessingError(RAGChatbotError):
    """Raised when document processing fails."""
    def __init__(self, document_path: str, stage: str, reason: str, original_error: Exception = None):
        super().__init__(
            message=f"Document processing failed at {stage} for '{document_path}': {reason}",
            error_code="DOCUMENT_PROCESSING_ERROR",
            context={
                "document_path": document_path,
                "processing_stage": stage,
                "reason": reason
            },
            original_error=original_error
        )


class DocumentLoadError(DocumentProcessingError):
    """Raised when a document cannot be loaded."""
    def __init__(self, document_path: str, reason: str, original_error: Exception = None):
        super().__init__(
            document_path=document_path,
            stage="loading",
            reason=reason,
            original_error=original_error
        )


class DocumentParsingError(DocumentProcessingError):
    """Raised when document content cannot be parsed."""
    def __init__(self, document_path: str, parser_type: str, reason: str, original_error: Exception = None):
        super().__init__(
            document_path=document_path,
            stage=f"parsing ({parser_type})",
            reason=reason,
            original_error=original_error
        )


class DocumentChunkingError(DocumentProcessingError):
    """Raised when document chunking fails."""
    def __init__(self, document_path: str, chunk_strategy: str, reason: str, original_error: Exception = None):
        super().__init__(
            document_path=document_path,
            stage=f"chunking ({chunk_strategy})",
            reason=reason,
            original_error=original_error
        )


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document is empty or contains no extractable content."""
    def __init__(self, document_path: str):
        super().__init__(
            document_path=document_path,
            stage="content extraction",
            reason="document is empty or contains no extractable text"
        )


# ============================================================================
# EMBEDDING & VECTOR STORE ERRORS
# ============================================================================

class EmbeddingError(RAGChatbotError):
    """Base class for embedding-related errors."""
    def __init__(self, operation: str, reason: str, context: Dict[str, Any] = None, original_error: Exception = None):
        super().__init__(
            message=f"Embedding error during {operation}: {reason}",
            error_code="EMBEDDING_ERROR",
            context=context or {},
            original_error=original_error
        )


class EmbeddingModelError(EmbeddingError):
    """Raised when there's an issue with the embedding model."""
    def __init__(self, model_name: str, reason: str, original_error: Exception = None):
        super().__init__(
            operation="model initialization",
            reason=f"Model '{model_name}' - {reason}",
            context={"model_name": model_name},
            original_error=original_error
        )


class EmbeddingAPIError(EmbeddingError):
    """Raised when embedding API calls fail."""
    def __init__(self, api_provider: str, reason: str, retry_count: int = 0, original_error: Exception = None):
        super().__init__(
            operation="API call",
            reason=f"{api_provider} API - {reason}",
            context={
                "api_provider": api_provider,
                "retry_count": retry_count
            },
            original_error=original_error
        )


class EmbeddingCacheError(EmbeddingError):
    """Raised when embedding cache operations fail."""
    def __init__(self, operation: str, reason: str, cache_path: str = None, original_error: Exception = None):
        super().__init__(
            operation=f"cache {operation}",
            reason=reason,
            context={"cache_path": cache_path},
            original_error=original_error
        )


class VectorStoreError(RAGChatbotError):
    """Base class for vector store errors."""
    def __init__(self, operation: str, reason: str, store_type: str = None, context: Dict[str, Any] = None, original_error: Exception = None):
        super().__init__(
            message=f"Vector store error during {operation}: {reason}",
            error_code="VECTOR_STORE_ERROR",
            context=context or {},
            original_error=original_error
        )

        if store_type:
            self.context["store_type"] = store_type


class VectorIndexError(VectorStoreError):
    """Raised when vector index operations fail."""
    def __init__(self, operation: str, index_path: str, reason: str, original_error: Exception = None):
        super().__init__(
            operation=f"index {operation}",
            reason=reason,
            context={"index_path": index_path},
            original_error=original_error
        )


class VectorSearchError(VectorStoreError):
    """Raised when vector similarity search fails."""
    def __init__(self, query_vector_size: int, reason: str, search_params: Dict[str, Any] = None, original_error: Exception = None):
        super().__init__(
            operation="similarity search",
            reason=reason,
            context={
                "query_vector_size": query_vector_size,
                "search_params": search_params or {}
            },
            original_error=original_error
        )


# ============================================================================
# QUERY PROCESSING ERRORS
# ============================================================================

class QueryProcessingError(RAGChatbotError):
    """Base class for query processing errors."""
    def __init__(self, query: str, stage: str, reason: str, context: Dict[str, Any] = None, original_error: Exception = None):
        super().__init__(
            message=f"Query processing failed at {stage}: {reason}",
            error_code="QUERY_PROCESSING_ERROR",
            context={
                "query": query[:100] + "..." if len(query) > 100 else query,  # Truncate long queries
                "processing_stage": stage,
                **(context or {})
            },
            original_error=original_error
        )


class QueryValidationError(QueryProcessingError):
    """Raised when query validation fails."""
    def __init__(self, query: str, validation_issue: str):
        super().__init__(
            query=query,
            stage="validation",
            reason=validation_issue
        )


class QueryRetrievalError(QueryProcessingError):
    """Raised when document retrieval for a query fails."""
    def __init__(self, query: str, retrieval_method: str, reason: str, original_error: Exception = None):
        super().__init__(
            query=query,
            stage=f"retrieval ({retrieval_method})",
            reason=reason,
            context={"retrieval_method": retrieval_method},
            original_error=original_error
        )


class ResponseGenerationError(QueryProcessingError):
    """Raised when LLM response generation fails."""
    def __init__(self, query: str, llm_model: str, reason: str, context_length: int = None, original_error: Exception = None):
        super().__init__(
            query=query,
            stage="response generation",
            reason=f"{llm_model} - {reason}",
            context={
                "llm_model": llm_model,
                "context_length": context_length
            },
            original_error=original_error
        )


# ============================================================================
# WEB SEARCH ERRORS
# ============================================================================

class WebSearchError(RAGChatbotError):
    """Base class for web search errors."""
    def __init__(self, query: str, operation: str, reason: str, context: Dict[str, Any] = None, original_error: Exception = None):
        super().__init__(
            message=f"Web search error during {operation}: {reason}",
            error_code="WEB_SEARCH_ERROR",
            context={
                "query": query,
                "operation": operation,
                **(context or {})
            },
            original_error=original_error
        )


class WebSearchAPIError(WebSearchError):
    """Raised when web search API calls fail."""
    def __init__(self, query: str, api_provider: str, reason: str, status_code: int = None, original_error: Exception = None):
        super().__init__(
            query=query,
            operation="API call",
            reason=f"{api_provider} - {reason}",
            context={
                "api_provider": api_provider,
                "status_code": status_code
            },
            original_error=original_error
        )


class WebContentExtractionError(WebSearchError):
    """Raised when web content extraction fails."""
    def __init__(self, url: str, reason: str, original_error: Exception = None):
        super().__init__(
            query="N/A",
            operation="content extraction",
            reason=reason,
            context={"url": url},
            original_error=original_error
        )


class WebSearchTimeoutError(WebSearchError):
    """Raised when web search operations timeout."""
    def __init__(self, query: str, operation: str, timeout_seconds: int):
        super().__init__(
            query=query,
            operation=operation,
            reason=f"operation timed out after {timeout_seconds} seconds",
            context={"timeout_seconds": timeout_seconds}
        )


# ============================================================================
# API & EXTERNAL SERVICE ERRORS
# ============================================================================

class ExternalServiceError(RAGChatbotError):
    """Base class for external service errors."""
    def __init__(self, service: str, operation: str, reason: str, status_code: int = None, context: Dict[str, Any] = None, original_error: Exception = None):
        super().__init__(
            message=f"External service error - {service} {operation}: {reason}",
            error_code="EXTERNAL_SERVICE_ERROR",
            context={
                "service": service,
                "operation": operation,
                "status_code": status_code,
                **(context or {})
            },
            original_error=original_error
        )


class APIRateLimitError(ExternalServiceError):
    """Raised when API rate limits are exceeded."""
    def __init__(self, service: str, retry_after: int = None, daily_quota_exceeded: bool = False):
        reason = "rate limit exceeded"
        if daily_quota_exceeded:
            reason += " (daily quota reached)"
        elif retry_after:
            reason += f", retry after {retry_after} seconds"

        super().__init__(
            service=service,
            operation="API call",
            reason=reason,
            context={
                "retry_after": retry_after,
                "daily_quota_exceeded": daily_quota_exceeded
            }
        )


class APIAuthenticationError(ExternalServiceError):
    """Raised when API authentication fails."""
    def __init__(self, service: str, reason: str = "invalid or missing API key"):
        super().__init__(
            service=service,
            operation="authentication",
            reason=reason,
            status_code=401
        )


class APIQuotaExceededError(ExternalServiceError):
    """Raised when API usage quota is exceeded."""
    def __init__(self, service: str, quota_type: str, reset_time: str = None):
        reason = f"{quota_type} quota exceeded"
        if reset_time:
            reason += f", resets at {reset_time}"

        super().__init__(
            service=service,
            operation="quota check",
            reason=reason,
            context={
                "quota_type": quota_type,
                "reset_time": reset_time
            }
        )


# ============================================================================
# SYSTEM & RESOURCE ERRORS
# ============================================================================

class ResourceError(RAGChatbotError):
    """Base class for system resource errors."""
    def __init__(self, resource_type: str, operation: str, reason: str, context: Dict[str, Any] = None, original_error: Exception = None):
        super().__init__(
            message=f"Resource error - {resource_type} {operation}: {reason}",
            error_code="RESOURCE_ERROR",
            context={
                "resource_type": resource_type,
                "operation": operation,
                **(context or {})
            },
            original_error=original_error
        )


class MemoryError(ResourceError):
    """Raised when memory-related issues occur."""
    def __init__(self, operation: str, memory_usage_mb: float = None, available_mb: float = None):
        reason = "insufficient memory"
        context = {}
        if memory_usage_mb is not None:
            context["memory_usage_mb"] = memory_usage_mb
        if available_mb is not None:
            context["available_mb"] = available_mb

        super().__init__(
            resource_type="memory",
            operation=operation,
            reason=reason,
            context=context
        )


class DiskSpaceError(ResourceError):
    """Raised when disk space issues occur."""
    def __init__(self, operation: str, required_mb: float, available_mb: float, path: str = None):
        super().__init__(
            resource_type="disk space",
            operation=operation,
            reason=f"insufficient disk space (required: {required_mb}MB, available: {available_mb}MB)",
            context={
                "required_mb": required_mb,
                "available_mb": available_mb,
                "path": path
            }
        )


class NetworkError(ResourceError):
    """Raised when network connectivity issues occur."""
    def __init__(self, operation: str, host: str = None, reason: str = "network connectivity issue"):
        super().__init__(
            resource_type="network",
            operation=operation,
            reason=reason,
            context={"host": host}
        )


# ============================================================================
# VALIDATION & INPUT ERRORS
# ============================================================================

class ValidationError(RAGChatbotError):
    """Base class for input validation errors."""
    def __init__(self, input_type: str, input_value: Any, issue: str, expected: str = None):
        message = f"Invalid {input_type}: {issue}"
        if expected:
            message += f" (expected: {expected})"

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context={
                "input_type": input_type,
                "input_value": str(input_value)[:100],  # Truncate long values
                "issue": issue,
                "expected": expected
            }
        )


class InvalidParameterError(ValidationError):
    """Raised when invalid parameters are provided."""
    def __init__(self, parameter_name: str, value: Any, issue: str, valid_options: List[str] = None):
        expected = f"one of {valid_options}" if valid_options else None
        super().__init__(
            input_type=f"parameter '{parameter_name}'",
            input_value=value,
            issue=issue,
            expected=expected
        )


class InvalidQueryError(ValidationError):
    """Raised when a query is invalid or malformed."""
    def __init__(self, query: str, issue: str):
        super().__init__(
            input_type="query",
            input_value=query,
            issue=issue
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def handle_exception(e: Exception, context: str = None, logger = None) -> RAGChatbotError:
    """
    Convert any exception to a RAGChatbotError with proper context.

    Args:
        e: The original exception
        context: Additional context about where the error occurred
        logger: Logger instance for recording the error

    Returns:
        RAGChatbotError: Wrapped exception with context
    """
    if isinstance(e, RAGChatbotError):
        return e

    # Map common exceptions to specific error types
    error_mappings = {
        FileNotFoundError: lambda ex: FileDoesNotExist(str(ex), context),
        PermissionError: lambda ex: FilePermissionError(str(ex), context or "unknown operation"),
        ConnectionError: lambda ex: NetworkError(context or "network operation", reason=str(ex)),
        TimeoutError: lambda ex: WebSearchTimeoutError("unknown", context or "operation", 30),
        ValueError: lambda ex: ValidationError("input", str(ex), str(ex)),
        ImportError: lambda ex: DependencyError(str(ex), "import failed"),
    }

    # Get the appropriate error type
    error_class = error_mappings.get(type(e))
    if error_class:
        wrapped_error = error_class(e)
    else:
        # Generic wrapper for unknown exceptions
        wrapped_error = RAGChatbotError(
            message=f"Unexpected error{' in ' + context if context else ''}: {str(e)}",
            error_code="UNEXPECTED_ERROR",
            context={"context": context},
            original_error=e
        )

    if logger:
        logger.error(f"Exception handled: {wrapped_error.to_dict()}")

    return wrapped_error


def get_error_recovery_suggestions(error: RAGChatbotError) -> List[str]:
    """
    Get recovery suggestions for common errors.

    Args:
        error: The RAGChatbotError instance

    Returns:
        List[str]: List of recovery suggestions
    """

    suggestions = []

    if isinstance(error, (DirectoryNotFoundError, FileDoesNotExist)):
        suggestions.extend([
            "Check if the file/directory path is correct",
            "Ensure the file/directory exists",
            "Verify you have proper permissions to access the location"
        ])

    elif isinstance(error, APIAuthenticationError):
        suggestions.extend([
            "Verify your API key is correct and active",
            "Check if the API key has the required permissions",
            "Ensure the API key is properly set in environment variables"
        ])

    elif isinstance(error, APIRateLimitError):
        suggestions.extend([
            "Wait before retrying the request",
            "Consider implementing exponential backoff",
            "Check if you need to upgrade your API plan"
        ])

    elif isinstance(error, EmbeddingCacheError):
        suggestions.extend([
            "Try clearing the embedding cache",
            "Check if there's sufficient disk space",
            "Verify cache directory permissions"
        ])

    elif isinstance(error, MemoryError):
        suggestions.extend([
            "Process documents in smaller batches",
            "Clear unused variables and caches",
            "Consider increasing system memory"
        ])

    elif isinstance(error, FileTypeNotSupported):
        supported_types = error.context.get("supported_types", [])
        if supported_types:
            suggestions.append(f"Convert file to one of: {', '.join(supported_types)}")

    return suggestions
