import sqlite3
from datetime import datetime

DB_PATH = "dialogues.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dialogues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT,
        created_at TEXT,
        UNIQUE(user_id, name)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dialogue_id INTEGER,
        role TEXT CHECK(role IN ('user', 'assistant')),
        content TEXT,
        timestamp TEXT,
        FOREIGN KEY (dialogue_id) REFERENCES dialogues(id)
    )
    ''')

    conn.commit()
    conn.close()

def _get_or_create_dialogue_id(conn, user_id: int, name: str) -> int:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM dialogues WHERE user_id = ? AND name = ?",
        (user_id, name)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    now = '17:03'
    cursor.execute(
        "INSERT INTO dialogues (user_id, name, created_at) VALUES (?, ?, ?)",
        (user_id, name, now)
    )
    return cursor.lastrowid

def add_dialogue(user_id: int, name: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        dialogue_id = _get_or_create_dialogue_id(conn, user_id, name)
        conn.commit()
    finally:
        conn.close()
    return  

def add_phrase(user_id: int, name: str, user_msg: str, assistant_msg: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        dialogue_id = _get_or_create_dialogue_id(conn, user_id, name)
        now = datetime.now().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (dialogue_id, role, content, timestamp) VALUES (?, 'user', ?, ?)",
            (dialogue_id, user_msg, now)
        )
        cursor.execute(
            "INSERT INTO messages (dialogue_id, role, content, timestamp) VALUES (?, 'assistant', ?, ?)",
            (dialogue_id, assistant_msg, now)
        )
        conn.commit()
    finally:
        conn.close()

def get_user_dialogues(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM dialogues WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

def get_dialogue_created_at(user_id: int, name: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT created_at FROM dialogues WHERE user_id = ? AND name = ?",
            (user_id, name)
        )
        row = cursor.fetchone()
        return row
    finally:
        conn.close()
