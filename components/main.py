from embedder import prepare_document_chunks, create_embedded_chunks
from vector_store import load_faiss_index, save_faiss_index
from query_handler import search
from utils.constants import RAW_DOCS_DIRECTORY


def main():
    # Step 1: Load and chunk
    documents = prepare_document_chunks(RAW_DOCS_DIRECTORY)

    # Step 2: Embed
    vectors, metadata = create_embedded_chunks(documents)

    # Step 3: Save to disk
    save_faiss_index(vectors, metadata)

    # Step 4: Load index and search
    index, meta = load_faiss_index()

    results = search("What are my hobbies?", index, meta)

    for r in results:
        print(f"Found: {r['text']}\nFrom: {r['metadata']}\n")


if __name__ == "__main__":
    main()
