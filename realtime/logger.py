"""
系统日志模块
支持同时输出到控制台和文件，并按天滚动
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name="VideoPipe", log_dir="logs"):
    """
    配置全局日志，支持按天滚动
    """
    # 1. 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 2. 生成基础日志文件名
    log_file = log_path / "videopipe.log"

    # 3. 配置格式
    log_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除旧的 handler (防止重复打印)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 5. 文件输出 (按天滚动，保留30天)
    file_handler = TimedRotatingFileHandler(
        str(log_file), 
        when="midnight", 
        interval=1, 
        backupCount=30, 
        encoding='utf-8'
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # 6. 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

# 预设一个全局 logger 实例
logger = setup_logger()
