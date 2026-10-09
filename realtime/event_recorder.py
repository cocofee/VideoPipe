"""
事件记录器模块

保存过线事件截图和数据
"""

import cv2
import threading
import queue
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from .detector import CrossingEvent, BibStatus
from .database import Database


class EventRecorder:
    """
    事件记录器

    异步保存截图和数据到数据库
    """

    def __init__(self, output_dir: str, database: Database):
        """
        初始化事件记录器

        Args:
            output_dir: 输出目录
            database: 数据库实例
        """
        self.output_dir = Path(output_dir)
        self.database = database

        # 创建截图目录
        self.screenshot_full_dir = self.output_dir / "screenshots_full"
        self.screenshot_crop_dir = self.output_dir / "screenshots_crop"
        self.screenshot_bib_dir = self.output_dir / "screenshots_bib"

        self.screenshot_full_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_crop_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_bib_dir.mkdir(parents=True, exist_ok=True)

        # 事件队列
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 回调
        self._on_event_saved: Optional[Callable[[int], None]] = None

        # 统计
        self._saved_count = 0

    def set_on_event_saved(self, callback: Callable[[int], None]):
        """设置事件保存完成回调"""
        self._on_event_saved = callback

    def start(self):
        """启动记录器"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"[EventRecorder] 已启动，输出目录: {self.output_dir}")

    def stop(self):
        """停止记录器"""
        self._running = False
        if self._thread:
            self._queue.put(None)  # 发送停止信号
            self._thread.join(timeout=5)
            self._thread = None
        print(f"[EventRecorder] 已停止，共保存 {self._saved_count} 个事件")

    def record(self, event: CrossingEvent):
        """
        记录事件（异步）

        Args:
            event: 过线事件
        """
        if self._running:
            self._queue.put(event)

    def _process_loop(self):
        """处理循环"""
        while self._running:
            try:
                event = self._queue.get(timeout=1)
                if event is None:
                    break
                self._save_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[EventRecorder] 处理错误: {e}")

    def _save_event(self, event: CrossingEvent):
        """保存事件"""
        try:
            # 生成文件名时间部分
            time_for_filename = event.cross_time_str.replace(":", "-").replace(".", "-")
            bib_text = event.bib_number if event.bib_number else "unknown"

            screenshot_full = ""
            screenshot_crop = ""
            screenshot_bib = ""

            # 保存完整画面截图
            if event.frame is not None:
                full_filename = f"cross_{event.event_id:03d}_ID{event.track_id}_{time_for_filename}_full.jpg"
                full_path = self.screenshot_full_dir / full_filename

                # 在画面上绘制标注
                frame_annotated = event.frame.copy()
                x1, y1, x2, y2 = event.bbox
                center_x, bottom_y = event.position

                # 绘制终点线（这里用默认位置，实际应该从配置获取）
                # cv2.line(frame_annotated, (0, 400), (1920, 400), (0, 0, 255), 2)

                # 绘制骑手框
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame_annotated, (center_x, bottom_y), 10, (0, 255, 0), -1)

                # 绘制信息
                info_text = f"Rank:{event.rank} ID:{event.track_id} BIB:{bib_text} Time:{event.cross_time_str}"
                cv2.putText(frame_annotated, info_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                cv2.imwrite(str(full_path), frame_annotated)
                screenshot_full = str(full_path)

            # 保存骑手裁剪截图
            if event.crop is not None:
                crop_filename = f"cross_{event.event_id:03d}_ID{event.track_id}_{time_for_filename}_crop.jpg"
                crop_path = self.screenshot_crop_dir / crop_filename

                # 在裁剪图上添加信息
                crop_annotated = event.crop.copy()
                label = f"BIB:{bib_text} Rank:{event.rank}"
                cv2.putText(crop_annotated, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imwrite(str(crop_path), crop_annotated)
                screenshot_crop = str(crop_path)

            # 保存号码牌截图
            if event.bib_crop is not None:
                bib_filename = f"bib_{event.event_id:03d}_ID{event.track_id}_{time_for_filename}.jpg"
                bib_path = self.screenshot_bib_dir / bib_filename
                cv2.imwrite(str(bib_path), event.bib_crop)
                screenshot_bib = str(bib_path)

            # 保存到数据库
            db_event = {
                'event_id': event.event_id,
                'rank': event.rank,
                'track_id': event.track_id,
                'cross_time': event.cross_time,
                'cross_time_str': event.cross_time_str,
                'bib_number': event.bib_number,
                'bib_confidence': event.bib_confidence,
                'bib_status': event.bib_status.value,
                'detection_confidence': event.detection_confidence,
                'position_x': event.position[0],
                'position_y': event.position[1],
                'bbox': list(event.bbox),
                'screenshot_full': screenshot_full,
                'screenshot_crop': screenshot_crop,
                'screenshot_bib': screenshot_bib,
            }

            self.database.insert_event(db_event)
            self._saved_count += 1

            # 打印信息
            status_symbol = {
                BibStatus.RECOGNIZED: "✓",
                BibStatus.NEEDS_REVIEW: "?",
                BibStatus.UNRECOGNIZED: "✗"
            }.get(event.bib_status, "?")

            print(f"[EventRecorder] 保存事件 #{event.event_id}: "
                  f"BIB={bib_text} [{status_symbol}] Time={event.cross_time_str}")

            # 触发回调
            if self._on_event_saved:
                try:
                    self._on_event_saved(event.event_id)
                except Exception:
                    pass

        except Exception as e:
            print(f"[EventRecorder] 保存失败: {e}")

    @property
    def saved_count(self) -> int:
        """已保存事件数"""
        return self._saved_count

    @property
    def pending_count(self) -> int:
        """待处理队列长度"""
        return self._queue.qsize()


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("  EventRecorder 模块")
    print("=" * 50)
    print()
    print("此模块需要配合Detector使用")
    print("请运行 realtime/main.py 进行完整测试")
