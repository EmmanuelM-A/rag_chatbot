import sqlite3
import datetime
from utils.logger import get_logger
from config import QA_SQLITE_DB_PATH

def init_db():
    conn = sqlite3.connect(QA_SQLITE_DB_PATH)

    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            answer TEXT,
            sources TEXT
        )
    ''')

    conn.commit()

    conn.close()


def log_qa_pair(query, answer, sources):
    conn = sqlite3.connect(QA_SQLITE_DB_PATH)

    cursor = conn.cursor()

    timestamp = datetime.datetime.now().isoformat()

    cursor.execute("INSERT INTO qa_logs (timestamp, query, answer, sources) VALUES (?, ?, ?, ?)",
                   (timestamp, query, answer, str(sources))) # Store sources as string

    conn.commit()

    conn.close()


def view_qa_logs(db_path=QA_SQLITE_DB_PATH):
    """
    Connects to the SQLite database and prints all entries from the qa_logs table.

    Args:
        db_path (str): The path to the SQLite database file.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Select all columns from the qa_logs table
        cursor.execute("SELECT id, timestamp, query, answer, sources FROM qa_logs ORDER BY timestamp DESC")

        # Fetch all results
        rows = cursor.fetchall()

        if not rows:
            print(f"No entries found in the 'qa_logs' table in {db_path}.")
            return

        print(f"\n--- Q&A Logs from {db_path} ---")
        for row in rows:
            # Unpack the row for better readability
            log_id, timestamp, query, answer, sources = row
            print(f"ID: {log_id}")
            print(f"Timestamp: {timestamp}")
            print(f"Query: {query}")
            print(f"Answer: {answer}")
            print(f"Sources: {sources}")
            print("-" * 30) # Separator for clarity

    except sqlite3.Error as e:
        print(f"An error occurred while connecting to or querying the database: {e}")
    finally:
        if conn:
            conn.close()
