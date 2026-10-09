"""
数据库模块

SQLite实时存储过线事件
"""

import sqlite3
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class Database:
    """
    SQLite数据库管理

    线程安全，支持实时写入
    """

    def __init__(self, db_path: str):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._lock = threading.Lock()

        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 过线事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crossing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER UNIQUE,
                    rank INTEGER,
                    track_id INTEGER,
                    cross_time REAL,
                    cross_time_str TEXT,
                    bib_number TEXT,
                    bib_confidence REAL,
                    bib_status TEXT,
                    detection_confidence REAL,
                    position_x INTEGER,
                    position_y INTEGER,
                    bbox TEXT,
                    screenshot_full TEXT,
                    screenshot_crop TEXT,
                    screenshot_bib TEXT,
                    created_at TEXT,
                    modified_at TEXT,
                    is_void INTEGER DEFAULT 0,
                    notes TEXT
                )
            ''')

            # 选手名单表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS athletes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bib_number TEXT UNIQUE,
                    name TEXT,
                    team TEXT,
                    category TEXT
                )
            ''')

            # 芯片成绩表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chip_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bib_number TEXT UNIQUE,
                    name TEXT,
                    category TEXT,
                    finish_time TEXT
                )
            ''')

            # 配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')

            conn.commit()
            conn.close()

    def insert_event(self, event: Dict[str, Any]) -> int:
        """
        插入过线事件

        Args:
            event: 事件字典

        Returns:
            插入的记录ID
        """
        now = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO crossing_events (
                    event_id, rank, track_id, cross_time, cross_time_str,
                    bib_number, bib_confidence, bib_status, detection_confidence,
                    position_x, position_y, bbox,
                    screenshot_full, screenshot_crop, screenshot_bib,
                    created_at, modified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.get('event_id'),
                event.get('rank'),
                event.get('track_id'),
                event.get('cross_time'),
                event.get('cross_time_str'),
                event.get('bib_number'),
                event.get('bib_confidence'),
                event.get('bib_status'),
                event.get('detection_confidence'),
                event.get('position_x'),
                event.get('position_y'),
                json.dumps(event.get('bbox', [])),
                event.get('screenshot_full', ''),
                event.get('screenshot_crop', ''),
                event.get('screenshot_bib', ''),
                now,
                now
            ))

            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return record_id

    def update_event(self, event_id: int, updates: Dict[str, Any]) -> bool:
        """
        更新过线事件

        Args:
            event_id: 事件ID
            updates: 要更新的字段

        Returns:
            是否更新成功
        """
        if not updates:
            return False

        # 构建UPDATE语句
        set_parts = []
        values = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            values.append(value)

        set_parts.append("modified_at = ?")
        values.append(datetime.now().isoformat())
        values.append(event_id)

        sql = f"UPDATE crossing_events SET {', '.join(set_parts)} WHERE event_id = ?"

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql, values)
            affected = cursor.rowcount
            conn.commit()
            conn.close()

            return affected > 0

    def get_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        """获取单个事件"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM crossing_events WHERE event_id = ?', (event_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return dict(row)
            return None

    def get_all_events(self, include_void: bool = False) -> List[Dict[str, Any]]:
        """
        获取所有事件

        Args:
            include_void: 是否包含已作废的事件

        Returns:
            事件列表
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if include_void:
                cursor.execute('SELECT * FROM crossing_events ORDER BY rank')
            else:
                cursor.execute('SELECT * FROM crossing_events WHERE is_void = 0 ORDER BY rank')

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

    def get_event_count(self) -> int:
        """获取事件数量"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM crossing_events WHERE is_void = 0')
            count = cursor.fetchone()[0]
            conn.close()
            return count

    def get_latest_event_id(self) -> int:
        """获取最新的event_id"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT MAX(event_id) FROM crossing_events')
            result = cursor.fetchone()[0]
            conn.close()
            return result if result else 0

    def void_event(self, event_id: int) -> bool:
        """作废事件"""
        return self.update_event(event_id, {'is_void': 1})

    def unvoid_event(self, event_id: int) -> bool:
        """取消作废"""
        return self.update_event(event_id, {'is_void': 0})

    def set_config(self, key: str, value: str):
        """设置配置"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)
            ''', (key, value))
            conn.commit()
            conn.close()

    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """获取配置"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM config WHERE key = ?', (key,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else default

    def clear_events(self):
        """清空所有事件（慎用）"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM crossing_events')
            conn.commit()
            conn.close()


# 测试代码
if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 50)
    print("  Database 测试")
    print("=" * 50)
    print()

    # 创建临时数据库
    db_path = os.path.join(tempfile.gettempdir(), "test_timing.db")
    print(f"[信息] 数据库路径: {db_path}")

    db = Database(db_path)

    # 测试插入
    print("\n[测试] 插入事件...")
    event1 = {
        'event_id': 1,
        'rank': 1,
        'track_id': 101,
        'cross_time': 123.456,
        'cross_time_str': '02:03.456',
        'bib_number': '42',
        'bib_confidence': 0.95,
        'bib_status': 'recognized',
        'detection_confidence': 0.88,
        'position_x': 960,
        'position_y': 400,
        'bbox': [900, 300, 1020, 500],
    }
    record_id = db.insert_event(event1)
    print(f"  插入成功，ID: {record_id}")

    # 测试查询
    print("\n[测试] 查询事件...")
    retrieved = db.get_event(1)
    print(f"  event_id: {retrieved['event_id']}")
    print(f"  bib_number: {retrieved['bib_number']}")
    print(f"  cross_time_str: {retrieved['cross_time_str']}")

    # 测试更新
    print("\n[测试] 更新事件...")
    db.update_event(1, {'bib_number': '43', 'notes': '人工修改'})
    updated = db.get_event(1)
    print(f"  bib_number: {updated['bib_number']}")
    print(f"  notes: {updated['notes']}")

    # 测试统计
    print("\n[测试] 统计...")
    print(f"  事件数量: {db.get_event_count()}")
    print(f"  最新event_id: {db.get_latest_event_id()}")

    print("\n[完成] 数据库测试通过")

    # 清理
    os.remove(db_path)
