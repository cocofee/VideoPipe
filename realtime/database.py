"""
数据库模块

SQLite实时存储过线事件
"""

import sqlite3
import threading
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

try:
    from .logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("Database")


class Database:
    """
    SQLite数据库管理

    线程安全，支持实时写入
    使用持久连接以提高性能
    """

    def __init__(self, db_path: str):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._lock = threading.RLock()  # 使用RLock允许同线程重入
        self._read_lock = threading.RLock()
        self._ocr_lock = threading.RLock()

        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建持久连接（关键优化：避免每次操作都新建连接）
        # check_same_thread=False 允许跨线程使用，由 _lock 保证线程安全
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._configure_connection(self._conn)

        self._read_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        self._read_conn.row_factory = sqlite3.Row
        self._configure_connection(self._read_conn)

        self._ocr_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        self._ocr_conn.row_factory = sqlite3.Row
        self._configure_connection(self._ocr_conn)

        # 初始化数据库
        self._init_db()

    def _configure_connection(self, conn: sqlite3.Connection):
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA wal_autocheckpoint = 1000;")
        except Exception as e:
            logger.warning(f"[Database] 配置 SQLite 连接失败: {e}")

    def _get_conn(self):
        """获取数据库连接（复用持久连接）"""
        return self._conn

    def _get_read_conn(self):
        return self._read_conn

    def _get_ocr_conn(self):
        return self._ocr_conn

    def close(self):
        """关闭数据库连接"""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
        with self._read_lock:
            if self._read_conn:
                self._read_conn.close()
                self._read_conn = None
        with self._ocr_lock:
            if self._ocr_conn:
                self._ocr_conn.close()
                self._ocr_conn = None

    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.executescript('''
            CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
            
            CREATE TABLE IF NOT EXISTS crossing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER UNIQUE,
                rank INTEGER,
                track_id INTEGER,
                participant_id TEXT,
                raw_track_ids_json TEXT,
                sport_profile TEXT DEFAULT 'cycling',
                passage_index INTEGER DEFAULT 1,
                cross_time REAL,
                cross_time_str TEXT,
                cross_realtime TEXT,
                finish_time TEXT,
                bib_number TEXT,
                original_bib TEXT,
                bib_confidence REAL,
                bib_status TEXT,
                detection_confidence REAL,
                position_x INTEGER,
                position_y INTEGER,
                bbox TEXT,
                screenshot_full TEXT,
                screenshot_clean TEXT,
                screenshot_crop TEXT,
                screenshot_bib TEXT,
                source_id INTEGER DEFAULT 0,
                is_test INTEGER DEFAULT 0,
                is_void INTEGER DEFAULT 0,
                manual_corrected INTEGER DEFAULT 0,
                is_ai_correction INTEGER DEFAULT 0,
                evidence_dir TEXT,
                ocr_state TEXT DEFAULT "PENDING",
                created_at TEXT,
                modified_at TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS athletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bib_number TEXT UNIQUE,
                name TEXT,
                team TEXT,
                category TEXT
            );

            CREATE TABLE IF NOT EXISTS chip_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bib_number TEXT UNIQUE,
                name TEXT,
                category TEXT,
                finish_time TEXT
            );

            CREATE TABLE IF NOT EXISTS start_times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT UNIQUE,
                start_time TEXT
            );

            CREATE TABLE IF NOT EXISTS event_evidences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                source_id INTEGER,
                screenshot_full TEXT,
                screenshot_clean TEXT,
                screenshot_crop TEXT,
                screenshot_bib TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY (event_id) REFERENCES crossing_events (event_id) ON DELETE CASCADE
            );
        ''')
            conn.commit()

        # 2. 增量字段检查 (适配旧版本数据库)
        columns = {
            'cross_realtime': 'TEXT',
            'finish_time': 'TEXT',
            'is_test': 'INTEGER DEFAULT 0',
            'source_id': 'INTEGER DEFAULT 0',
            'manual_corrected': 'INTEGER DEFAULT 0',
            'original_bib': 'TEXT',
            'is_ai_correction': 'INTEGER DEFAULT 0',
            'evidence_dir': 'TEXT',
            'ocr_state': 'TEXT DEFAULT "PENDING"',
            'participant_id': 'TEXT',
            'raw_track_ids_json': 'TEXT',
            'sport_profile': "TEXT DEFAULT 'cycling'",
            'passage_index': 'INTEGER DEFAULT 1'
        }
        
        # 获取现有字段名
        cursor.execute("PRAGMA table_info(crossing_events)")
        existing_cols = [row[1] for row in cursor.fetchall()]
        
        for col_name, col_type in columns.items():
            if col_name not in existing_cols:
                try:
                    cursor.execute(f'ALTER TABLE crossing_events ADD COLUMN {col_name} {col_type}')
                except sqlite3.OperationalError:
                    pass
        
        conn.commit()

    def _normalize_bib(self, bib: Any) -> Optional[str]:
        """统一号码格式"""
        if bib is None:
            return None
        s = str(bib).strip()
        if not s or s.lower() == 'none' or s.lower() == 'null' or s.lower() == 'nan':
            return None
            
        # 处理可能的 .0 (来自Excel)
        try:
            if '.' in s:
                # 尝试转换为整数再转回字符串，以去掉 .0 或其他小数位
                return str(int(float(s)))
        except (ValueError, TypeError):
            pass
            
        return s

    @staticmethod
    def _normalize_raw_track_ids(raw_track_ids: Any, compatibility_track_id: Any = None) -> List[int]:
        value = raw_track_ids
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                value = []
        if value is None:
            value = []
        elif not isinstance(value, (list, tuple, set)):
            value = [value]

        normalized = set()
        for track_id in value:
            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                continue
            if track_id >= 0:
                normalized.add(track_id)

        if not normalized:
            try:
                compatibility_track_id = int(compatibility_track_id)
            except (TypeError, ValueError):
                compatibility_track_id = None
            if compatibility_track_id is not None and compatibility_track_id >= 0:
                normalized.add(compatibility_track_id)

        return sorted(normalized)

    def _event_record(self, row: sqlite3.Row) -> Dict[str, Any]:
        record = dict(row)
        record['raw_track_ids'] = self._normalize_raw_track_ids(
            record.get('raw_track_ids_json'),
            record.get('track_id'),
        )
        return record

    def _coerce_float(self, value: Any) -> Optional[float]:
        """将值转换为 float（用于相对时间秒）"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def is_bib_exists(self, bib_number: str, exclude_event_id: Optional[int] = None) -> bool:
        """
        检查号码是否已存在于有效记录中
        """
        return self.get_event_by_bib(bib_number, exclude_event_id) is not None

    def is_bib_exists_recent(self, bib_number: str, seconds: int = 30, exclude_event_id: Optional[int] = None, current_time: Optional[float] = None) -> bool:
        """
        检查号码在指定时间范围内（秒）是否已存在有效记录

        用于基于时间的去重，允许同一号码在一段时间后再次过线

        Args:
            bib_number: 号码
            seconds: 时间范围（秒），默认30秒
            exclude_event_id: 排除的事件ID

        Returns:
            是否存在近期的重复记录
        """
        if not bib_number or bib_number.strip() == "":
            return False

        bib = self._normalize_bib(bib_number)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            current = self._coerce_float(current_time)
            if current is None:
                cursor.execute("SELECT MAX(cross_time) FROM crossing_events WHERE is_void = 0")
                row = cursor.fetchone()
                current = self._coerce_float(row[0]) if row else None

            if current is None:
                return False

            threshold = current - seconds
            query = "SELECT * FROM crossing_events WHERE bib_number = ? AND is_void = 0 AND cross_time >= ?"
            params = [bib, threshold]

            if exclude_event_id is not None:
                query += " AND event_id != ?"
                params.append(exclude_event_id)

            query += " ORDER BY cross_time DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()

            return dict(row) is not None if row else False

    def get_event_by_bib(self, bib_number: str, exclude_event_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        获取具有指定号码的第一个有效记录
        """
        if not bib_number or bib_number.strip() == "":
            return None

        bib = self._normalize_bib(bib_number)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            query = "SELECT * FROM crossing_events WHERE bib_number = ? AND is_void = 0"
            params = [bib]

            if exclude_event_id is not None:
                query += " AND event_id != ?"
                params.append(exclude_event_id)

            query += " ORDER BY cross_time DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()

            return dict(row) if row else None

    def insert_evidence(self, evidence: Dict[str, Any]) -> int:
        """
        插入过线证据（多机位截图关联）

        Args:
            evidence: 证据数据
        """
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO event_evidences (
                    event_id, source_id, screenshot_full, screenshot_clean, 
                    screenshot_crop, screenshot_bib, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evidence.get('event_id'),
                evidence.get('source_id'),
                evidence.get('screenshot_full'),
                evidence.get('screenshot_clean'),
                evidence.get('screenshot_crop'),
                evidence.get('screenshot_bib'),
                evidence.get('confidence'),
                now
            ))
            record_id = cursor.lastrowid
            conn.commit()
            return record_id

    def upsert_best_evidence(self, event_id: int, evidence: Dict[str, Any]) -> int:
        now = datetime.now().isoformat()
        new_conf = evidence.get('confidence')
        if new_conf is None:
            new_conf = 0.0
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM event_evidences WHERE event_id = ? ORDER BY confidence DESC, id ASC', (event_id,))
            rows = cursor.fetchall()
            if not rows:
                cursor.execute('''
                    INSERT INTO event_evidences (
                        event_id, source_id, screenshot_full, screenshot_clean, 
                        screenshot_crop, screenshot_bib, confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_id,
                    evidence.get('source_id'),
                    evidence.get('screenshot_full'),
                    evidence.get('screenshot_clean'),
                    evidence.get('screenshot_crop'),
                    evidence.get('screenshot_bib'),
                    new_conf,
                    now
                ))
                record_id = cursor.lastrowid
                conn.commit()
                return record_id
            
            best = dict(rows[0])
            best_conf = best.get('confidence') if best.get('confidence') is not None else 0.0
            should_replace = new_conf > best_conf
            if not should_replace:
                if (not best.get('screenshot_bib')) and evidence.get('screenshot_bib'):
                    should_replace = True
                elif (not best.get('screenshot_crop')) and evidence.get('screenshot_crop'):
                    should_replace = True
                elif (not best.get('screenshot_full')) and evidence.get('screenshot_full'):
                    should_replace = True
                elif (not best.get('screenshot_clean')) and evidence.get('screenshot_clean'):
                    should_replace = True
            
            if should_replace:
                cursor.execute('''
                    UPDATE event_evidences
                    SET source_id = ?, screenshot_full = ?, screenshot_clean = ?, screenshot_crop = ?,
                        screenshot_bib = ?, confidence = ?, created_at = ?
                    WHERE id = ?
                ''', (
                    evidence.get('source_id'),
                    evidence.get('screenshot_full'),
                    evidence.get('screenshot_clean'),
                    evidence.get('screenshot_crop'),
                    evidence.get('screenshot_bib'),
                    new_conf,
                    now,
                    best.get('id')
                ))
                keep_id = best.get('id')
            else:
                keep_id = best.get('id')
            
            cursor.execute('DELETE FROM event_evidences WHERE event_id = ? AND id != ?', (event_id, keep_id))
            conn.commit()
            return keep_id

    def get_event_evidences(self, event_id: int) -> List[Dict[str, Any]]:
        """获取事件的所有关联证据"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM event_evidences WHERE event_id = ? ORDER BY source_id', (event_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def find_match_event(self, bib_number: Optional[str], cross_time: float, window_seconds: float = 2.0) -> Optional[Dict[str, Any]]:
        """
        寻找匹配的已存在事件（用于合并）

        Args:
            bib_number: 号码（可选）
            cross_time: 过线时间（相对秒）
            window_seconds: 时间窗口
        """
        cross = self._coerce_float(cross_time)
        if cross is None:
            return None

        if not bib_number or bib_number.strip() == "" or bib_number.upper() == "UNKNOWN":
            # 如果没有号码，通过时间窗口寻找最近的一个记录
            # 注意：无号码合并风险较高，仅限非常短的时间窗口
            with self._lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                
                # 计算窗口
                t_min = cross - (window_seconds / 2)
                t_max = cross + (window_seconds / 2)
                
                cursor.execute('''
                    SELECT * FROM crossing_events 
                    WHERE cross_time BETWEEN ? AND ? 
                    AND is_void = 0 
                    AND (bib_number IS NULL OR bib_number = '' OR bib_number = 'UNKNOWN')
                    ORDER BY ABS(cross_time - ?) ASC
                    LIMIT 1
                ''', (t_min, t_max, cross))
                row = cursor.fetchone()
                return dict(row) if row else None
        else:
            # 有号码，在较大窗口内寻找同一号码
            return self.get_event_by_bib_and_time(bib_number, cross, window_seconds=window_seconds)

    def get_event_by_bib_and_time(self, bib_number: str, cross_time: float, window_seconds: float = 30.0) -> Optional[Dict[str, Any]]:
        """根据号码和时间寻找匹配事件（相对秒）"""
        bib = self._normalize_bib(bib_number)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cross = self._coerce_float(cross_time)
            if cross is None:
                return None

            t_min = cross - window_seconds
            t_max = cross + window_seconds
            
            cursor.execute('''
                SELECT * FROM crossing_events 
                WHERE bib_number = ? AND is_void = 0 
                AND cross_time BETWEEN ? AND ?
                ORDER BY ABS(cross_time - ?) ASC, cross_time DESC, event_id DESC
                LIMIT 1
            ''', (bib, t_min, t_max, cross))
            row = cursor.fetchone()
            return dict(row) if row else None

    def check_duplicate_record(self, source_id: int, time_bucket: float, bib_number: str) -> bool:
        """
        检查是否存在重复记录 (弱唯一约束)
        
        Args:
            source_id: 摄像头 ID
            time_bucket: 时间桶（秒级精度，单位：相对秒）
            bib_number: 号码牌
        """
        if not bib_number or bib_number.upper() == "UNKNOWN":
            return False
            
        bib = self._normalize_bib(bib_number)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # 搜索该秒内的记录
            t_start = float(time_bucket)
            t_end = t_start + 1.0
            
            cursor.execute('''
                SELECT id FROM crossing_events 
                WHERE source_id = ? AND bib_number = ? AND is_void = 0
                AND cross_time >= ? AND cross_time < ?
                LIMIT 1
            ''', (source_id, bib, t_start, t_end))
            return cursor.fetchone() is not None

    def insert_event(self, event: Dict[str, Any], enable_dedup: bool = True) -> int:
        """
        插入过线事件

        Args:
            event: 事件数据
            enable_dedup: 是否检查号码重复（默认True）
        """
        now = datetime.now().isoformat()
        bib = self._normalize_bib(event.get('bib_number'))

        # 方案1: 检查号码是否已存在且未作废（仅在启用去重时）
        # 仅对非空号码进行检查
        if enable_dedup and bib and bib.strip() != "":
            if self.is_bib_exists(bib):
                logger.info(f"[Database] 号码 {bib} 已在数据库中存在，跳过重复插入")
                return -2  # 使用特定错误码表示号码已存在

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO crossing_events (
                    event_id, rank, track_id, participant_id, raw_track_ids_json, sport_profile, passage_index,
                    cross_time, cross_time_str,
                    cross_realtime, bib_number, original_bib, bib_confidence, bib_status, detection_confidence,
                    source_id, position_x, position_y, bbox,
                    screenshot_full, screenshot_clean, screenshot_crop, screenshot_bib,
                    is_test, finish_time, evidence_dir, ocr_state, notes,
                    created_at, modified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.get('event_id'),
                event.get('rank'),
                event.get('track_id'),
                event.get('participant_id') or None,
                json.dumps(
                    self._normalize_raw_track_ids(
                        event.get('raw_track_ids', event.get('raw_track_ids_json')),
                        event.get('track_id'),
                    )
                ),
                event.get('sport_profile') or 'cycling',
                int(event.get('passage_index') or 1),
                event.get('cross_time'),
                event.get('cross_time_str'),
                event.get('cross_realtime'),
                bib,
                event.get('original_bib', bib),
                event.get('bib_confidence'),
                event.get('bib_status'),
                event.get('detection_confidence'),
                event.get('source_id', 0),
                event.get('position_x'),
                event.get('position_y'),
                json.dumps(event.get('bbox', [])),
                event.get('screenshot_full', ''),
                event.get('screenshot_clean', ''),
                event.get('screenshot_crop', ''),
                event.get('screenshot_bib', ''),
                event.get('is_test', 0),
                event.get('finish_time'),
                event.get('evidence_dir', ''),
                event.get('ocr_state', 'PENDING'),
                event.get('notes', ''),
                now,
                now
            ))

            record_id = cursor.lastrowid
            conn.commit()

            return record_id

    def update_event(self, event_id: int, updates: Dict[str, Any]) -> bool:
        """
        更新过线事件

        Args:
            event_id: 事件ID
            updates: 要更新的字段
        """
        if not updates:
            return False

        # 检查是否为 AI 自动修正
        is_ai = updates.pop('is_ai_correction', False)
        if is_ai:
            updates['is_ai_correction'] = 1

        # 构建UPDATE语句
        set_parts = []
        values = []
        
        # 特殊处理号码修改：标记为人工修正（除非是 AI 修正）
        if 'bib_number' in updates and not is_ai:
            set_parts.append("manual_corrected = 1")
        
        for key, value in updates.items():
            if key == 'bib_number':
                value = self._normalize_bib(value)
            set_parts.append(f"{key} = ?")
            values.append(value)

        set_parts.append("modified_at = ?")
        values.append(datetime.now().isoformat())
        values.append(event_id)

        sql = f"UPDATE crossing_events SET {', '.join(set_parts)} WHERE event_id = ?"

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(sql, values)
            affected = cursor.rowcount
            conn.commit()

            return affected > 0

    def update_event_ocr_result(self, event_id: int, bib: str, conf: float, ocr_state: str, evidence_dir: str = "", notes: str = "", use_ocr_conn: bool = False) -> bool:
        """更新 OCR 识别结果及状态"""
        lock = self._ocr_lock if use_ocr_conn else self._lock
        conn = self._get_ocr_conn() if use_ocr_conn else self._get_conn()

        for attempt in range(6):
            with lock:
                try:
                    cursor = conn.cursor()

                    now = datetime.now().isoformat()
                    bib_norm = self._normalize_bib(bib)
                    if bib_norm:
                        bib_status = 'recognized' if (conf or 0.0) >= 0.4 else 'needs_review'
                    else:
                        bib_status = 'unrecognized'

                    if bib_norm:
                        if evidence_dir:
                            cursor.execute('''
                                UPDATE crossing_events 
                                SET bib_number = ?, bib_confidence = ?, bib_status = ?, ocr_state = ?, evidence_dir = ?, modified_at = ?, notes = COALESCE(notes, '') || ' ' || ?
                                WHERE event_id = ?
                            ''', (bib_norm, conf, bib_status, ocr_state, evidence_dir, now, notes, event_id))
                        else:
                            cursor.execute('''
                                UPDATE crossing_events 
                                SET bib_number = ?, bib_confidence = ?, bib_status = ?, ocr_state = ?, modified_at = ?, notes = COALESCE(notes, '') || ' ' || ?
                                WHERE event_id = ?
                            ''', (bib_norm, conf, bib_status, ocr_state, now, notes, event_id))
                    else:
                        if evidence_dir:
                            cursor.execute('''
                                UPDATE crossing_events 
                                SET bib_number = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN NULL ELSE bib_number END,
                                    bib_confidence = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN 0.0 ELSE bib_confidence END,
                                    bib_status = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN ? ELSE bib_status END,
                                    ocr_state = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN ? ELSE ocr_state END,
                                    evidence_dir = ?, modified_at = ?, notes = COALESCE(notes, '') || ' ' || ?
                                WHERE event_id = ?
                            ''', (bib_status, ocr_state, evidence_dir, now, notes, event_id))
                        else:
                            cursor.execute('''
                                UPDATE crossing_events 
                                SET bib_number = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN NULL ELSE bib_number END,
                                    bib_confidence = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN 0.0 ELSE bib_confidence END,
                                    bib_status = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN ? ELSE bib_status END,
                                    ocr_state = CASE WHEN COALESCE(manual_corrected, 0) = 0 THEN ? ELSE ocr_state END,
                                    modified_at = ?, notes = COALESCE(notes, '') || ' ' || ?
                                WHERE event_id = ?
                            ''', (bib_status, ocr_state, now, notes, event_id))

                    conn.commit()
                    return True
                except sqlite3.OperationalError as e:
                    msg = str(e).lower()
                    if "locked" in msg or "busy" in msg:
                        pass
                    else:
                        logger.error(f"[Database] 更新 OCR 结果失败 {event_id}: {e}")
                        return False
                except Exception as e:
                    logger.error(f"[Database] 更新 OCR 结果失败 {event_id}: {e}")
                    return False

            time.sleep(min(0.5, 0.05 * (2 ** attempt)))

        logger.error(f"[Database] 更新 OCR 结果失败 {event_id}: database locked")
        return False

    def merge_ocr_duplicate_event(
        self,
        duplicate_event_id: int,
        target_event_id: int,
        bib: str,
        conf: float,
        ocr_state: str,
        evidence_dir: str = "",
        notes: str = "",
        use_ocr_conn: bool = False,
    ) -> bool:
        """Merge an OCR-confirmed duplicate while preserving its evidence."""
        if int(duplicate_event_id) == int(target_event_id):
            return False

        bib_norm = self._normalize_bib(bib)
        if not bib_norm:
            return False

        lock = self._ocr_lock if use_ocr_conn else self._lock
        conn = self._get_ocr_conn() if use_ocr_conn else self._get_conn()

        for attempt in range(6):
            with lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT * FROM crossing_events WHERE event_id IN (?, ?)",
                        (int(target_event_id), int(duplicate_event_id)),
                    )
                    rows = {int(row["event_id"]): dict(row) for row in cursor.fetchall()}
                    target = rows.get(int(target_event_id))
                    duplicate = rows.get(int(duplicate_event_id))
                    if not target or not duplicate or int(target.get("is_void") or 0) != 0:
                        return False

                    try:
                        target_conf = float(target.get("bib_confidence") or 0.0)
                    except (TypeError, ValueError):
                        target_conf = 0.0
                    merged_conf = max(target_conf, float(conf or 0.0))
                    bib_status = "recognized" if merged_conf >= 0.4 else "needs_review"
                    now = datetime.now().isoformat()

                    target_notes = str(target.get("notes") or "").strip()
                    merge_note = f"MERGED_FROM:{int(duplicate_event_id)}"
                    target_note_parts = [part for part in (target_notes, notes, merge_note) if part]
                    merged_target_notes = " ".join(target_note_parts)

                    def prefer_target(field: str):
                        return target.get(field) or duplicate.get(field) or ""

                    cursor.execute(
                        '''
                        UPDATE crossing_events
                        SET bib_number = ?, original_bib = COALESCE(NULLIF(original_bib, ''), ?),
                            bib_confidence = ?, bib_status = ?, ocr_state = ?,
                            screenshot_full = ?, screenshot_clean = ?, screenshot_crop = ?, screenshot_bib = ?,
                            evidence_dir = ?, modified_at = ?, notes = ?
                        WHERE event_id = ?
                        ''',
                        (
                            bib_norm,
                            bib_norm,
                            merged_conf,
                            bib_status,
                            ocr_state,
                            prefer_target("screenshot_full"),
                            prefer_target("screenshot_clean"),
                            prefer_target("screenshot_crop"),
                            prefer_target("screenshot_bib"),
                            target.get("evidence_dir") or evidence_dir or duplicate.get("evidence_dir") or "",
                            now,
                            merged_target_notes,
                            int(target_event_id),
                        ),
                    )

                    cursor.execute(
                        "UPDATE event_evidences SET event_id = ? WHERE event_id = ?",
                        (int(target_event_id), int(duplicate_event_id)),
                    )

                    duplicate_notes = str(duplicate.get("notes") or "").strip()
                    duplicate_merge_note = f"MERGED_TO:{int(target_event_id)}"
                    if duplicate_merge_note not in duplicate_notes:
                        duplicate_notes = f"{duplicate_notes} {duplicate_merge_note}".strip()
                    cursor.execute(
                        '''
                        UPDATE crossing_events
                        SET bib_number = ?, bib_confidence = ?, bib_status = ?, ocr_state = ?,
                            is_void = 1, modified_at = ?, notes = ?
                        WHERE event_id = ?
                        ''',
                        (
                            bib_norm,
                            float(conf or 0.0),
                            bib_status,
                            ocr_state,
                            now,
                            duplicate_notes,
                            int(duplicate_event_id),
                        ),
                    )
                    conn.commit()
                    return True
                except sqlite3.OperationalError as e:
                    conn.rollback()
                    if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                        logger.error(f"[Database] OCR duplicate merge failed: {e}")
                        return False
                except Exception as e:
                    conn.rollback()
                    logger.error(f"[Database] OCR duplicate merge failed: {e}")
                    return False

            time.sleep(min(0.5, 0.05 * (2 ** attempt)))

        logger.error(
            f"[Database] OCR duplicate merge failed: database locked, "
            f"duplicate={duplicate_event_id}, target={target_event_id}"
        )
        return False

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """获取单个事件（关联选手和芯片成绩）"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            query = '''
                SELECT
                    e.*,
                    a.name as athlete_name,
                    a.category as athlete_category,
                    a.team as athlete_team,
                    c.finish_time as chip_time
                FROM crossing_events e
                LEFT JOIN athletes a ON e.bib_number = a.bib_number
                LEFT JOIN chip_results c ON e.bib_number = c.bib_number
                WHERE e.event_id = ?
            '''
            cursor.execute(query, (event_id,))
            row = cursor.fetchone()

            if row:
                return self._event_record(row)
            return None

    def get_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        """兼容旧调用：按 event_id 获取事件。"""
        return self.get_event_by_id(event_id)

    def get_all_events(self, include_void: bool = False) -> List[Dict[str, Any]]:
        """获取所有事件（关联选手信息）"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            query = '''
                SELECT
                    e.*,
                    a.name as athlete_name,
                    a.category as athlete_category,
                    a.team as athlete_team,
                    c.finish_time as chip_time
                FROM crossing_events e
                LEFT JOIN athletes a ON e.bib_number = a.bib_number
                LEFT JOIN chip_results c ON e.bib_number = c.bib_number
            '''

            if not include_void:
                query += ' WHERE e.is_void = 0'

            query += ' ORDER BY e.event_id ASC'

            cursor.execute(query)
            rows = cursor.fetchall()
            return [self._event_record(row) for row in rows]

    def get_event_count(self) -> int:
        """获取事件数量"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM crossing_events WHERE is_void = 0')
            count = cursor.fetchone()[0]
            return count

    def get_latest_event_id(self) -> int:
        """获取最新的event_id"""
        with self._read_lock:
            conn = self._get_read_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT MAX(event_id) FROM crossing_events')
            result = cursor.fetchone()[0]
            return result if result else 0

    def get_latest_cross_time(self) -> Optional[float]:
        with self._read_lock:
            conn = self._get_read_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(cross_time) FROM crossing_events WHERE is_void = 0")
            row = cursor.fetchone()
            if not row:
                return None
            try:
                return float(row[0]) if row[0] is not None else None
            except (TypeError, ValueError):
                return None

    def get_latest_event_by_track_id(self, track_id: int, source_id: int) -> Optional[int]:
        """根据 track_id 和 source_id 获取最近的一个有效事件 ID（限制在近期窗口内，避免track_id复用误更新）"""
        with self._read_lock:
            conn = self._get_read_conn()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT event_id FROM crossing_events 
                WHERE track_id = ? AND source_id = ? AND is_void = 0
                  AND cross_time >= (SELECT COALESCE(MAX(cross_time), 0) FROM crossing_events WHERE is_void = 0) - ?
                ORDER BY event_id DESC LIMIT 1
            ''', (track_id, source_id, 10.0))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_events_since(self, last_id: int) -> List[Dict[str, Any]]:
        """获取指定ID之后的所有事件（关联选手信息）"""
        with self._read_lock:
            conn = self._get_read_conn()
            cursor = conn.cursor()

            query = '''
                SELECT
                    e.*,
                    a.name as athlete_name,
                    a.category as athlete_category,
                    a.team as athlete_team,
                    c.finish_time as chip_time
                FROM crossing_events e
                LEFT JOIN athletes a ON e.bib_number = a.bib_number
                LEFT JOIN chip_results c ON e.bib_number = c.bib_number
                WHERE e.event_id > ?
                ORDER BY e.event_id ASC
            '''

            cursor.execute(query, (last_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_events_since_time(self, seconds: float = 1.0, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """获取最近指定秒数内的所有事件（相对时间）"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            current = self._coerce_float(current_time)
            if current is None:
                cursor.execute("SELECT MAX(cross_time) FROM crossing_events WHERE is_void = 0")
                row = cursor.fetchone()
                current = self._coerce_float(row[0]) if row else None

            if current is None:
                return []

            threshold = current - seconds

            query = '''
                SELECT * FROM crossing_events 
                WHERE cross_time >= ? AND is_void = 0
                ORDER BY cross_time DESC
            '''
            cursor.execute(query, (threshold,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def has_recent_unknown_event(self, seconds: float = 1.0, current_time: Optional[float] = None) -> bool:
        """检查最近是否有一个未识别号码的记录（用于多机位去重，基于相对时间）"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            current = self._coerce_float(current_time)
            if current is None:
                cursor.execute("SELECT MAX(cross_time) FROM crossing_events WHERE is_void = 0")
                row = cursor.fetchone()
                current = self._coerce_float(row[0]) if row else None

            if current is None:
                return False

            threshold = current - seconds

            query = "SELECT COUNT(*) FROM crossing_events WHERE (bib_number IS NULL OR bib_number = '' OR bib_number = 'UNKNOWN') AND is_void = 0 AND cross_time >= ?"
            cursor.execute(query, (threshold,))
            count = cursor.fetchone()[0]
            return count > 0

    def get_modified_events_since(self, since_time: str) -> List[Dict[str, Any]]:
        """获取在指定时间之后被修改的所有事件"""
        with self._read_lock:
            conn = self._get_read_conn()
            cursor = conn.cursor()

            query = '''
                SELECT
                    e.*,
                    a.name as athlete_name,
                    a.category as athlete_category,
                    a.team as athlete_team,
                    c.finish_time as chip_time
                FROM crossing_events e
                LEFT JOIN athletes a ON e.bib_number = a.bib_number
                LEFT JOIN chip_results c ON e.bib_number = c.bib_number
                WHERE e.modified_at > ?
                ORDER BY e.modified_at ASC
            '''

            cursor.execute(query, (since_time,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def void_event(self, event_id: int) -> bool:
        """作废事件"""
        return self.update_event(event_id, {'is_void': 1})

    def unvoid_event(self, event_id: int) -> bool:
        """取消作废"""
        return self.update_event(event_id, {'is_void': 0})

    def update_event_bib(self, event_id: int, bib_number: str, is_ai: bool = False) -> bool:
        """
        更新过线事件的号码
        is_ai: 是否由 AI 修正
        """
        now = datetime.now().isoformat()
        bib = self._normalize_bib(bib_number)

        with self._lock:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()

                bib_status = 'recognized' if bib else 'unrecognized'

                if is_ai:
                    cursor.execute('''
                        UPDATE crossing_events
                        SET bib_number = ?, bib_status = ?, is_ai_correction = 1, modified_at = ?
                        WHERE event_id = ?
                    ''', (bib, bib_status, now, event_id))
                else:
                    cursor.execute('''
                        UPDATE crossing_events
                        SET bib_number = ?, bib_status = ?, is_ai_correction = 0, manual_corrected = 1, modified_at = ?
                        WHERE event_id = ?
                    ''', (bib, bib_status, now, event_id))

                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Error updating bib: {e}")
                return False

    def set_config(self, key: str, value: str):
        """设置配置"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)
            ''', (key, value))
            conn.commit()

    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """获取配置"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM config WHERE key = ?', (key,))
            row = cursor.fetchone()
            return row[0] if row else default

    def upsert_athletes(self, athletes_list: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        增量更新选手名单 (Upsert)

        Args:
            athletes_list: 包含选手信息的列表，每个元素为字典:
                {'bib_number': '1', 'name': '张三', 'category': '男子组', 'team': '某某车队'}

        Returns:
            (新增数量, 更新数量)
        """
        added = 0
        updated = 0
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            for athlete in athletes_list:
                bib = self._normalize_bib(athlete.get('bib_number'))
                if not bib:
                    continue

                # 检查是否存在
                cursor.execute('SELECT id FROM athletes WHERE bib_number = ?', (bib,))
                row = cursor.fetchone()

                if row:
                    # 更新
                    cursor.execute('''
                        UPDATE athletes SET name = ?, category = ?, team = ?
                        WHERE bib_number = ?
                    ''', (athlete.get('name', ''), athlete.get('category', ''), athlete.get('team', ''), bib))
                    updated += 1
                else:
                    # 插入
                    cursor.execute('''
                        INSERT INTO athletes (bib_number, name, category, team)
                        VALUES (?, ?, ?, ?)
                    ''', (bib, athlete.get('name', ''), athlete.get('category', ''), athlete.get('team', '')))
                    added += 1

            conn.commit()
        return added, updated

    def get_athlete_by_bib(self, bib_number: str) -> Optional[Dict[str, Any]]:
        """根据号码获取选手信息"""
        bib = self._normalize_bib(bib_number)
        if not bib:
            return None

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM athletes WHERE bib_number = ?', (bib,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_all_athletes(self) -> List[Dict[str, Any]]:
        """获取所有选手信息"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM athletes')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_athlete_summary(self) -> Dict[str, Any]:
        """
        获取选手名单的统计特征，用于校验识别结果
        返回: {
            'bibs': set(),           # 所有号码集合
            'max_numeric': int,      # 纯数字号码的最大值
            'allowed_chars': set(),  # 出现过的所有字符（字母/数字）
            'has_alpha': bool        # 是否包含带字母的号码
        }
        """
        athletes = self.get_all_athletes()
        bibs = set()
        max_numeric = 0
        allowed_chars = set()
        has_alpha = False

        for a in athletes:
            bib = a.get('bib_number')
            if not bib:
                continue
            
            bib_str = str(bib).strip().upper()
            bibs.add(bib_str)
            
            # 统计允许的字符
            for char in bib_str:
                allowed_chars.add(char)
            
            # 检查是否含字母
            if any(c.isalpha() for c in bib_str):
                has_alpha = True
            
            # 统计最大数字
            if bib_str.isdigit():
                max_numeric = max(max_numeric, int(bib_str))
        
        # 获取配置中的号码段规则
        bib_ranges_str = self.get_config('bib_ranges', '')
        
        return {
            'bibs': bibs,
            'max_numeric': max_numeric,
            'allowed_chars': allowed_chars,
            'has_alpha': has_alpha,
            'bib_ranges_str': bib_ranges_str
        }

    def upsert_chip_results(self, chip_list: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        增量更新芯片成绩 (Upsert)

        Args:
            chip_list: 每个元素为字典:
                {'bib_number': '1', 'finish_time': '10:00:01.123', 'name': '张三', 'category': '男子组'}
        """
        added = 0
        updated = 0
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            for item in chip_list:
                bib = self._normalize_bib(item.get('bib_number'))
                if not bib:
                    continue

                cursor.execute('SELECT id FROM chip_results WHERE bib_number = ?', (bib,))
                row = cursor.fetchone()

                if row:
                    cursor.execute('''
                        UPDATE chip_results SET finish_time = ?, name = ?, category = ?
                        WHERE bib_number = ?
                    ''', (item.get('finish_time', ''), item.get('name', ''), item.get('category', ''), bib))
                    updated += 1
                else:
                    cursor.execute('''
                        INSERT INTO chip_results (bib_number, finish_time, name, category)
                        VALUES (?, ?, ?, ?)
                    ''', (bib, item.get('finish_time', ''), item.get('name', ''), item.get('category', '')))
                    added += 1

            conn.commit()
        return added, updated

    def get_all_categories(self) -> List[str]:
        """获取所有已存在的选手分组"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT category FROM athletes WHERE category IS NOT NULL AND category != ""')
            categories = [row[0] for row in cursor.fetchall()]
            return sorted(categories)

    def get_export_data(self, include_void: bool = False) -> List[Dict[str, Any]]:
        """
        获取导出数据（关联选手名单和芯片成绩）
        """
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            query = '''
                SELECT
                    e.*,
                    a.name as athlete_name,
                    a.category as athlete_category,
                    a.team as athlete_team,
                    c.finish_time as chip_time
                FROM crossing_events e
                LEFT JOIN athletes a ON e.bib_number = a.bib_number
                LEFT JOIN chip_results c ON e.bib_number = c.bib_number
            '''

            if not include_void:
                query += ' WHERE e.is_void = 0'

            query += ' ORDER BY e.rank ASC'

            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def clear_events(self):
        """清空所有事件（慎用）"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM event_evidences')
            cursor.execute('DELETE FROM crossing_events')
            conn.commit()

    # ========== 发枪时间相关 ==========

    def set_start_time(self, category: str, start_time: str) -> bool:
        """
        设置某个组别的发枪时间

        Args:
            category: 组别名称
            start_time: 发枪时间（格式：HH:MM:SS）
        """
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO start_times (category, start_time)
                VALUES (?, ?)
            ''', (category, start_time))
            conn.commit()
            return True

    def get_start_time(self, category: str) -> Optional[str]:
        """获取某个组别的发枪时间"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT start_time FROM start_times WHERE category = ?', (category,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_start_times(self) -> List[Dict[str, str]]:
        """获取所有组别的发枪时间（返回列表）"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT category, start_time FROM start_times')
            return [{'category': row[0], 'start_time': row[1]} for row in cursor.fetchall()]

    def delete_start_time(self, category: str) -> bool:
        """删除某个组别的发枪时间"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM start_times WHERE category = ?', (category,))
            conn.commit()
            return cursor.rowcount > 0

    def get_categories_without_start_time(self) -> List[str]:
        """获取没有设置发枪时间的组别"""
        all_categories = self.get_all_categories()
        start_times = self.get_all_start_times()
        start_time_dict = {st['category']: st['start_time'] for st in start_times}
        return [cat for cat in all_categories if cat not in start_time_dict or not start_time_dict[cat]]

    # ========== 测试模式相关 ==========

    def get_is_test_mode(self) -> bool:
        """获取当前是否测试模式"""
        value = self.get_config('is_test_mode', '1')  # 默认测试模式
        return value == '1'

    def set_is_test_mode(self, is_test: bool):
        """设置测试模式"""
        self.set_config('is_test_mode', '1' if is_test else '0')

    def clear_test_events(self) -> int:
        """清除所有测试数据，返回删除的数量"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM crossing_events WHERE is_test = 1')
            count = cursor.fetchone()[0]
            cursor.execute('DELETE FROM crossing_events WHERE is_test = 1')
            conn.commit()
            return count

    def get_test_event_count(self) -> int:
        """获取测试数据数量"""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM crossing_events WHERE is_test = 1')
            return cursor.fetchone()[0]

    # ========== 成绩计算相关 ==========

    def calculate_finish_time(self, cross_realtime: Any, category: str) -> Optional[str]:
        """
        计算成绩

        Args:
            cross_realtime: 过线时间（相对秒或时间字符串）
            category: 组别

        Returns:
            成绩字符串（格式：H:MM:SS.mmm）或 None
        """
        start_time_str = self.get_start_time(category)
        if not start_time_str:
            return None

        try:
            cross_seconds = self._coerce_float(cross_realtime)
            if cross_seconds is not None:
                if cross_seconds > 1e12:
                    cross_seconds = cross_seconds / 1000.0
                if cross_seconds > 1e9:
                    cross_dt = datetime.fromtimestamp(cross_seconds)
                    if len(start_time_str) > 10:
                        start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    else:
                        cross_date = cross_dt.strftime('%Y-%m-%d')
                        start_dt = datetime.strptime(f"{cross_date} {start_time_str}", '%Y-%m-%d %H:%M:%S')

                    diff = cross_dt - start_dt
                    if diff.total_seconds() < 0:
                        diff = diff + timedelta(days=1)

                    total_seconds = diff.total_seconds()
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = total_seconds % 60
                    return f"{hours}:{minutes:02d}:{seconds:06.3f}"

                start_seconds = self._coerce_float(start_time_str)
                if start_seconds is None:
                    start_seconds = 0.0
                diff_seconds = cross_seconds - start_seconds
                if diff_seconds < 0:
                    diff_seconds = cross_seconds
                hours = int(diff_seconds // 3600)
                minutes = int((diff_seconds % 3600) // 60)
                seconds = diff_seconds % 60
                return f"{hours}:{minutes:02d}:{seconds:06.3f}"

            from datetime import datetime, timedelta

            cross_str = str(cross_realtime).strip()
            if '-' in cross_str or 'T' in cross_str:
                cross_str = cross_str.replace('T', ' ')
                if '.' in cross_str:
                    cross_dt = datetime.strptime(cross_str, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    cross_dt = datetime.strptime(cross_str, '%Y-%m-%d %H:%M:%S')
            else:
                time_fmt = '%H:%M:%S.%f' if '.' in cross_str else '%H:%M:%S'
                cross_time_only = datetime.strptime(cross_str, time_fmt).time()
                if len(start_time_str) > 10:
                    base_date = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S').date()
                else:
                    base_date = datetime.now().date()
                cross_dt = datetime.combine(base_date, cross_time_only)

            # 解析发枪时间（可能是完整格式或仅时间格式）
            if len(start_time_str) > 10:  # 完整格式 YYYY-MM-DD HH:MM:SS
                start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
            else:
                cross_date = cross_dt.strftime('%Y-%m-%d')
                start_dt = datetime.strptime(f"{cross_date} {start_time_str}", '%Y-%m-%d %H:%M:%S')

            diff = cross_dt - start_dt
            if diff.total_seconds() < 0:
                diff = diff + timedelta(days=1)

            total_seconds = diff.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60

            return f"{hours}:{minutes:02d}:{seconds:06.3f}"

        except Exception as e:
            logger.error(f"[Database] 计算成绩失败: {e}")
            return None

    def update_event_finish_time(self, event_id: int) -> bool:
        """
        根据事件的过线时间和组别，重新计算并更新成绩和过线时间显示
        
        用于人工修改号码后自动更新成绩
        """
        event = self.get_event_by_id(event_id)
        if not event:
            return False

        bib_number = event.get('bib_number')
        cross_time = event.get('cross_time')
        cross_realtime = event.get('cross_realtime')

        if not bib_number or (cross_time is None and not cross_realtime):
            return False

        # 获取组别
        athlete = self.get_athlete_by_bib(bib_number)
        if not athlete:
            # 如果没找到选手，尝试用默认发枪时间更新 cross_time_str
            all_starts = self.get_all_start_times()
            if all_starts:
                ref_cat = all_starts[0]['category']
                new_time = self.calculate_finish_time(cross_time if cross_time is not None else cross_realtime, ref_cat)
                if new_time:
                    return self.update_event(event_id, {'cross_time_str': new_time, 'finish_time': None})
            return False

        category = athlete.get('category')
        if not category:
            return False

        # 计算成绩
        finish_time = self.calculate_finish_time(cross_time if cross_time is not None else cross_realtime, category)
        if not finish_time:
            return False

        # 更新成绩和过线时间显示（使其保持一致）
        return self.update_event(event_id, {
            'finish_time': finish_time,
            'cross_time_str': finish_time
        })

    # ========== 赛事初始化与归档相关 ==========

    def reset_business_data(self) -> bool:
        """
        清空所有业务数据（记录、选手、成绩、发枪时间）
        保留配置（如摄像头、终点线位置）
        """
        with self._lock:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute('DELETE FROM crossing_events')
                cursor.execute('DELETE FROM athletes')
                cursor.execute('DELETE FROM chip_results')
                cursor.execute('DELETE FROM start_times')
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"[Database] 清空业务数据失败: {e}")
                return False

    def get_hard_samples(self) -> List[Dict[str, Any]]:
        """
        获取难点照片记录：
        1. 人工修改过号码的记录 (manual_corrected = 1)
        2. AI未识别出号码的记录 (bib_number 为空)
        """
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM crossing_events 
                WHERE manual_corrected = 1 
                OR bib_number IS NULL 
                OR bib_number = ""
                ORDER BY cross_time DESC
            '''
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

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
