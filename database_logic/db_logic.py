import sqlite3
from datetime import datetime
from typing import List

DB_PATH = "dialogues.db"

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dialogues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT,
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

def _get_or_create_dialogue_id(conn: sqlite3.Connection, user_id: int, name: str) -> int:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM dialogues WHERE user_id = ? AND name = ?",
        (user_id, name)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(
        "INSERT INTO dialogues (user_id, name) VALUES (?, ?)",
        (user_id, name)
    )
    return cursor.lastrowid

def add_dialogue(user_id: int, name: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        _get_or_create_dialogue_id(conn, user_id, name)
        conn.commit()
    finally:
        conn.close()

def add_phrase(user_id: int, name: str, user_msg: str, assistant_msg: str) -> None:
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

def get_user_dialogues(user_id: int) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM dialogues WHERE user_id = ?",
            (user_id,)
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

def get_full_dialogue(user_id: int, name: str) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM dialogues WHERE user_id = ? AND name = ?",
            (user_id, name)
        )
        row = cursor.fetchone()
        if not row:
            return [] 
        dialogue_id = row[0]

        cursor.execute(
            "SELECT role, content FROM messages "
            "WHERE dialogue_id = ? "
            "ORDER BY id ASC",
            (dialogue_id,)
        )
        result = []
        for role, content in cursor.fetchall():
            prefix = "Assistant" if role == "assistant" else "User"
            if content == None:
                content = ''
            elif content[10::] == 'Assistant: ':
                content[10::]
            result.append(f"{prefix}: {content}")
        return result
    finally:
        conn.close()

def delete_dialogue(user_id: int, name: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        # Найдём id диалога
        cursor.execute(
            "SELECT id FROM dialogues WHERE user_id = ? AND name = ?",
            (user_id, name)
        )
        row = cursor.fetchone()
        if not row:
            return False

        dialogue_id = row[0]

        cursor.execute(
            "DELETE FROM messages WHERE dialogue_id = ?",
            (dialogue_id,)
        )
        cursor.execute(
            "DELETE FROM dialogues WHERE id = ?",
            (dialogue_id,)
        )

        conn.commit()
        return True
    finally:
        conn.close()

