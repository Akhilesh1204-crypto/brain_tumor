import sqlite3
import json
from datetime import datetime

DB_PATH = 'scans.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            heatmap_path TEXT,
            raw_class TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            metadata TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_scan(case_id, filename, file_path, heatmap_path, raw_class, class_name, confidence, probabilities, metadata=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO scans (case_id, created_at, filename, file_path, heatmap_path, raw_class, class_name, confidence, probabilities, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        case_id,
        now_str,
        filename,
        file_path,
        heatmap_path,
        raw_class,
        class_name,
        confidence,
        json.dumps(probabilities),
        json.dumps(metadata) if metadata else None
    ))
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def get_history(limit=25):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scans ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        results.append({
            'id': r['id'],
            'case_id': r['case_id'],
            'created_at': r['created_at'],
            'filename': r['filename'],
            'file_path': r['file_path'],
            'heatmap_path': r['heatmap_path'],
            'raw_class': r['raw_class'],
            'class_name': r['class_name'],
            'confidence': r['confidence'],
            'probabilities': json.loads(r['probabilities']) if r['probabilities'] else {},
            'metadata': json.loads(r['metadata']) if r['metadata'] else {}
        })
    conn.close()
    return results

def delete_scan(scan_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

init_db()
