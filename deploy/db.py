"""
检测结果数据库模块
使用 SQLite 存储所有检测记录，支持查询、统计、趋势分析

功能：
    - 检测历史存储与查询
    - 缺陷严重程度分级
    - 报警阈值按类别可调
    - 最近打开文件记录
    - 趋势统计
"""

import sqlite3
import os
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

DB_PATH = Path(__file__).parent / "data" / "detections.db"


def _ensure_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            image_name TEXT,
            class_name TEXT,
            class_name_cn TEXT,
            confidence REAL,
            severity TEXT,
            severity_score REAL DEFAULT 0.0,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            inference_time_ms REAL,
            has_defect INTEGER,
            defect_count INTEGER DEFAULT 0,
            annotated_path TEXT,
            source_type TEXT DEFAULT 'image'
        );

        CREATE TABLE IF NOT EXISTS recent_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            file_type TEXT DEFAULT 'image',
            last_opened TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alert_config (
            class_name TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL DEFAULT 1,
            threshold REAL NOT NULL DEFAULT 0.25,
            severity_levels TEXT DEFAULT '{"轻微":0.3,"中等":0.5,"严重":0.7}'
        );

        CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_class ON detections(class_name);
        CREATE INDEX IF NOT EXISTS idx_severity ON detections(severity);
        CREATE INDEX IF NOT EXISTS idx_has_defect ON detections(has_defect);

        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            original_class TEXT,
            original_class_cn TEXT,
            original_confidence REAL,
            reviewed_class TEXT,
            reviewed_class_cn TEXT,
            action TEXT,
            reviewer TEXT DEFAULT 'operator'
        );

        CREATE INDEX IF NOT EXISTS idx_review_timestamp ON reviews(timestamp);
    """)

    # 初始化默认报警配置
    default_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    for cls in default_classes:
        conn.execute(
            "INSERT OR IGNORE INTO alert_config (class_name, enabled, threshold) VALUES (?, 1, 0.25)",
            (cls,)
        )

    conn.commit()
    conn.close()


# ══════════════════════════════════════════════
# 严重程度分级
# ══════════════════════════════════════════════

# 默认分级阈值
DEFAULT_SEVERITY_RULES = {
    'crazing':         {'轻微': 0.25, '中等': 0.45, '严重': 0.65},
    'inclusion':       {'轻微': 0.25, '中等': 0.50, '严重': 0.70},
    'patches':         {'轻微': 0.25, '中等': 0.55, '严重': 0.75},
    'pitted_surface':  {'轻微': 0.25, '中等': 0.50, '严重': 0.70},
    'rolled-in_scale': {'轻微': 0.25, '中等': 0.50, '严重': 0.70},
    'scratches':       {'轻微': 0.25, '中等': 0.55, '严重': 0.75},
}

# 严重程度颜色
SEVERITY_COLORS = {
    '危急': '#ff0000',
    '严重': '#f87171',
    '中等': '#fbbf24',
    '轻微': '#4ade80',
    '合格': '#4ade80',
}


def get_alert_threshold(class_name: str) -> float:
    """获取指定类别的报警阈值"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT threshold FROM alert_config WHERE class_name = ?", (class_name,)
        ).fetchone()
        return row['threshold'] if row else 0.25
    finally:
        conn.close()


def get_all_alert_configs() -> Dict[str, Dict]:
    """获取所有类别的报警配置"""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM alert_config").fetchall()
        configs = {}
        for row in rows:
            configs[row['class_name']] = {
                'enabled': bool(row['enabled']),
                'threshold': row['threshold'],
                'severity_levels': json.loads(row['severity_levels']) if row['severity_levels'] else {},
            }
        return configs
    finally:
        conn.close()


def update_alert_config(class_name: str, enabled: bool = None, threshold: float = None,
                        severity_levels: Dict = None):
    """更新类别报警配置"""
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT * FROM alert_config WHERE class_name = ?", (class_name,)
        ).fetchone()

        if existing:
            if enabled is not None:
                conn.execute("UPDATE alert_config SET enabled = ? WHERE class_name = ?",
                             (int(enabled), class_name))
            if threshold is not None:
                conn.execute("UPDATE alert_config SET threshold = ? WHERE class_name = ?",
                             (threshold, class_name))
            if severity_levels is not None:
                conn.execute("UPDATE alert_config SET severity_levels = ? WHERE class_name = ?",
                             (json.dumps(severity_levels, ensure_ascii=False), class_name))
        else:
            conn.execute(
                "INSERT INTO alert_config (class_name, enabled, threshold, severity_levels) VALUES (?,?,?,?)",
                (class_name, int(enabled or True), threshold or 0.25,
                 json.dumps(severity_levels or DEFAULT_SEVERITY_RULES.get(class_name, {}), ensure_ascii=False))
            )
        conn.commit()
    finally:
        conn.close()


# Severity 规则缓存（避免每帧每检测都查询数据库）
_severity_cache = {}  # {class_name: rules_dict}
_severity_cache_loaded = False


def _load_severity_cache():
    """加载所有 severity 规则到内存缓存"""
    global _severity_cache, _severity_cache_loaded
    if _severity_cache_loaded:
        return

    conn = get_connection()
    try:
        rows = conn.execute("SELECT class_name, severity_levels FROM alert_config").fetchall()
        for row in rows:
            if row['severity_levels']:
                _severity_cache[row['class_name']] = json.loads(row['severity_levels'])
    finally:
        conn.close()
    _severity_cache_loaded = True


def refresh_severity_cache():
    """刷新 severity 缓存（设置变更后调用）"""
    global _severity_cache_loaded
    _severity_cache_loaded = False
    _severity_cache.clear()
    _load_severity_cache()


def get_severity(class_name: str, confidence: float) -> tuple:
    """
    根据缺陷类别和置信度判断严重程度

    返回: (严重程度字符串, 严重程度分数 0-100)
    """
    # 使用内存缓存，避免每帧查询数据库
    _load_severity_cache()
    rules = _severity_cache.get(class_name,
                                DEFAULT_SEVERITY_RULES.get(class_name, {'轻微': 0.25, '中等': 0.50, '严重': 0.70}))

    severe_thresh = rules.get('严重', 0.70)
    medium_thresh = rules.get('中等', 0.50)
    light_thresh = rules.get('轻微', 0.25)

    if confidence >= severe_thresh:
        score = 70 + (confidence - severe_thresh) / (1.0 - severe_thresh) * 30
        return ('严重', round(min(score, 100), 1))
    elif confidence >= medium_thresh:
        score = 40 + (confidence - medium_thresh) / (severe_thresh - medium_thresh) * 30
        return ('中等', round(score, 1))
    elif confidence >= light_thresh:
        score = 10 + (confidence - light_thresh) / (medium_thresh - light_thresh) * 30
        return ('轻微', round(score, 1))
    else:
        score = confidence / light_thresh * 10
        return ('轻微', round(max(score, 0), 1))


def compute_overall_severity(detections: list) -> tuple:
    """
    计算整体严重程度（综合所有缺陷）

    考虑因素：
    - 缺陷数量
    - 最高置信度
    - 缺陷类别多样性

    返回: (整体严重程度, 整体分数)
    """
    if not detections:
        return ('合格', 0)

    severities = []
    for det in detections:
        sev, score = get_severity(det.class_name if hasattr(det, 'class_name') else det.get('class_name', ''),
                                  det.confidence if hasattr(det, 'confidence') else det.get('confidence', 0))
        severities.append((sev, score))

    max_score = max(s[1] for s in severities)
    defect_count = len(detections)
    unique_types = len(set(d.class_name if hasattr(d, 'class_name') else d.get('class_name', '') for d in detections))

    # 多缺陷叠加
    count_bonus = min(defect_count * 5, 20)
    diversity_bonus = min(unique_types * 3, 10)

    final_score = min(max_score + count_bonus + diversity_bonus, 100)

    if final_score >= 70:
        return ('严重', round(final_score, 1))
    elif final_score >= 40:
        return ('中等', round(final_score, 1))
    else:
        return ('轻微', round(final_score, 1))


def severity_color(severity: str) -> str:
    """严重程度对应颜色"""
    return SEVERITY_COLORS.get(severity, '#888888')


# ══════════════════════════════════════════════
# 写入检测记录
# ══════════════════════════════════════════════

def save_detection(image_path: str, detections: list, inference_time_ms: float,
                   source_type: str = 'image', annotated_path: str = None):
    """
    保存一次检测的所有缺陷到数据库

    参数:
        image_path: 原始图片路径
        detections: 检测结果列表 (SingleDetection 对象或 dict)
        inference_time_ms: 推理耗时
        source_type: 来源类型 (image/camera/video)
        annotated_path: 标注后图片保存路径
    """
    conn = get_connection()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_name = Path(image_path).name

    # 计算整体严重程度
    overall_severity, overall_score = compute_overall_severity(detections)

    for det in detections:
        # 兼容 SingleDetection 对象和 dict
        if hasattr(det, 'class_name'):
            cls_name = det.class_name
            cls_name_cn = det.class_name_cn
            conf = det.confidence
            bbox = det.bbox
        else:
            cls_name = det.get('class_name', '')
            cls_name_cn = det.get('class_name_cn', '')
            conf = det.get('confidence', 0)
            bbox = det.get('bbox', [0, 0, 0, 0])

        severity, sev_score = get_severity(cls_name, conf)

        conn.execute("""
            INSERT INTO detections
            (timestamp, image_path, image_name, class_name, class_name_cn, confidence,
             severity, severity_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             inference_time_ms, has_defect, defect_count, annotated_path, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, image_path, image_name, cls_name, cls_name_cn,
            conf, severity, sev_score,
            bbox[0], bbox[1], bbox[2], bbox[3],
            inference_time_ms, 1, len(detections), annotated_path, source_type
        ))

    # 如果没有缺陷，也记录一条
    if not detections:
        conn.execute("""
            INSERT INTO detections
            (timestamp, image_path, image_name, class_name, class_name_cn, confidence,
             severity, inference_time_ms, has_defect, defect_count, annotated_path, source_type)
            VALUES (?, ?, ?, NULL, NULL, NULL, '合格', ?, 0, 0, ?, ?)
        """, (timestamp, image_path, image_name, inference_time_ms, annotated_path, source_type))

    conn.commit()
    conn.close()
    return overall_severity, overall_score


# ══════════════════════════════════════════════
# 查询
# ══════════════════════════════════════════════

def get_recent(limit: int = 100) -> List[dict]:
    """获取最近的检测记录"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history(limit: int = 100, offset: int = 0, has_defect: Optional[bool] = None,
                severity: Optional[str] = None, class_name: Optional[str] = None,
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
    """查询检测历史（带筛选）"""
    conn = get_connection()
    try:
        conditions = []
        params = []

        if has_defect is not None:
            conditions.append("has_defect = ?")
            params.append(int(has_defect))
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if class_name:
            conditions.append("class_name = ?")
            params.append(class_name)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        rows = conn.execute(
            f"""SELECT * FROM detections WHERE {where_clause}
                ORDER BY id DESC LIMIT ? OFFSET ?""",
            params + [limit, offset]
        ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_stats(hours: int = 24) -> dict:
    """获取统计信息"""
    conn = get_connection()
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    total = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE timestamp >= ?", (since,)
    ).fetchone()[0]

    defects = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE has_defect=1 AND timestamp >= ?", (since,)
    ).fetchone()[0]

    # 各类别统计
    class_stats = conn.execute("""
        SELECT class_name_cn, COUNT(*) as cnt,
               AVG(confidence) as avg_conf
        FROM detections
        WHERE has_defect=1 AND timestamp >= ?
        GROUP BY class_name
        ORDER BY cnt DESC
    """, (since,)).fetchall()

    # 各严重程度统计
    severity_stats = conn.execute("""
        SELECT severity, COUNT(*) as cnt
        FROM detections
        WHERE has_defect=1 AND timestamp >= ?
        GROUP BY severity
    """, (since,)).fetchall()

    # 平均推理时间
    avg_time = conn.execute(
        "SELECT AVG(inference_time_ms) FROM detections WHERE timestamp >= ?", (since,)
    ).fetchone()[0] or 0

    conn.close()

    return {
        'total': total,
        'defects': defects,
        'pass_count': total - defects,
        'pass_rate': ((total - defects) / total * 100) if total > 0 else 0,
        'class_stats': [dict(r) for r in class_stats],
        'severity_stats': [dict(r) for r in severity_stats],
        'avg_inference_ms': round(avg_time, 1),
    }


def get_trend(hours: int = 24, interval_minutes: int = 60) -> List[dict]:
    """获取缺陷趋势（按时间范围自动调整分组粒度）"""
    conn = get_connection()
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    # 根据时间范围选择分组粒度
    if hours <= 24:
        # 24小时内：按小时
        fmt = '%Y-%m-%d %H:00:00'
    elif hours <= 168:
        # 7天内：按天
        fmt = '%Y-%m-%d'
    else:
        # 30天：按天
        fmt = '%Y-%m-%d'

    rows = conn.execute(f"""
        SELECT
            strftime('{fmt}', timestamp) as time_bucket,
            COUNT(*) as total,
            SUM(has_defect) as defects
        FROM detections
        WHERE timestamp >= ?
        GROUP BY time_bucket
        ORDER BY time_bucket
    """, (since,)).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def get_class_trend(hours: int = 24) -> List[dict]:
    """获取各类别缺陷趋势"""
    conn = get_connection()
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    if hours <= 24:
        fmt = '%Y-%m-%d %H:00:00'
    else:
        fmt = '%Y-%m-%d'

    rows = conn.execute(f"""
        SELECT
            strftime('{fmt}', timestamp) as time_bucket,
            class_name_cn,
            COUNT(*) as cnt
        FROM detections
        WHERE has_defect=1 AND timestamp >= ?
        GROUP BY time_bucket, class_name
        ORDER BY time_bucket
    """, (since,)).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def get_daily_stats(days: int = 30) -> List[dict]:
    """获取每日检测统计"""
    conn = get_connection()
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT
            DATE(timestamp) as date,
            COUNT(*) as total,
            SUM(has_defect) as defect_count,
            SUM(CASE WHEN has_defect = 0 THEN 1 ELSE 0 END) as pass_count
        FROM detections
        WHERE timestamp >= ?
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, (since,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════
# 最近文件管理
# ══════════════════════════════════════════════

def add_recent_file(file_path: str, file_type: str = "image"):
    """添加最近打开的文件"""
    conn = get_connection()
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 先删除已有记录
        conn.execute("DELETE FROM recent_files WHERE file_path = ?", (file_path,))
        conn.execute(
            "INSERT INTO recent_files (file_path, file_type, last_opened) VALUES (?, ?, ?)",
            (file_path, file_type, now)
        )
        # 只保留最近 20 条
        conn.execute("""
            DELETE FROM recent_files WHERE id NOT IN (
                SELECT id FROM recent_files ORDER BY last_opened DESC LIMIT 20
            )
        """)
        conn.commit()
    finally:
        conn.close()


def get_recent_files(limit: int = 10) -> List[dict]:
    """获取最近打开的文件"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM recent_files ORDER BY last_opened DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def clear_recent_files():
    """清空最近文件记录"""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM recent_files")
        conn.commit()
    finally:
        conn.close()


# ══════════════════════════════════════════════
# 数据清理
# ══════════════════════════════════════════════

def save_review(image_path: str, original_class: str, original_class_cn: str,
                original_confidence: float, reviewed_class: str, reviewed_class_cn: str,
                action: str):
    """保存人工复核结果"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO reviews (timestamp, image_path, original_class, original_class_cn, "
            "original_confidence, reviewed_class, reviewed_class_cn, action) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path,
             original_class, original_class_cn, original_confidence,
             reviewed_class, reviewed_class_cn, action)
        )
        conn.commit()
    finally:
        conn.close()


def get_review_stats() -> dict:
    """获取复核统计"""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        confirmed = conn.execute("SELECT COUNT(*) FROM reviews WHERE action='确认'").fetchone()[0]
        corrected = conn.execute("SELECT COUNT(*) FROM reviews WHERE action='修正'").fetchone()[0]
        return {'total': total, 'confirmed': confirmed, 'corrected': corrected}
    finally:
        conn.close()


def delete_detection(detection_id: int) -> bool:
    """删除单条检测记录"""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def cleanup_old_records(days: int = 90):
    """清理超过指定天数的旧记录"""
    conn = get_connection()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        deleted = conn.execute(
            "DELETE FROM detections WHERE timestamp < ?", (cutoff,)
        ).rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


# 初始化
init_db()
