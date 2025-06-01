class FileDocument:
    def __init__(self, content, metadata):
        """
        Represents a document with its content and metadata.

        :param content: The main content of the document (e.g., text from a file).
        :param metadata: A dictionary or object containing metadata about the document (e.g., filename, file extension,
        author, etc.).
        """
        self.content = content
        self.metadata = metadata
