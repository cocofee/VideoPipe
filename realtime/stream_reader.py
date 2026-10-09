"""
流读取模块

支持RTSP摄像头和USB摄像头的实时流读取
使用独立线程持续读帧，不阻塞主程序
"""

import cv2
import threading
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

try:
    from .independent_counting import FrameStamp
except ImportError:
    from independent_counting import FrameStamp
from typing import Optional, Tuple, Callable
from enum import Enum


class StreamStatus(Enum):
    """流状态枚举"""
    DISCONNECTED = "disconnected"  # 未连接
    CONNECTING = "connecting"      # 连接中
    CONNECTED = "connected"        # 已连接
    RECONNECTING = "reconnecting"  # 重连中
    ERROR = "error"                # 错误


class StreamReader:
    """
    流读取器

    支持RTSP流和USB摄像头，使用独立线程持续读帧

    用法:
        # RTSP流
        reader = StreamReader("rtsp://localhost:8554/test")

        # USB摄像头
        reader = StreamReader(0)  # 设备索引

        reader.start()
        while True:
            frame = reader.get_frame()
            if frame is not None:
                cv2.imshow("Preview", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        reader.stop()
    """

    def __init__(self, source, buffer_size: int = 1):
        """
        初始化流读取器

        Args:
            source: 流源，可以是:
                - RTSP URL字符串，如 "rtsp://localhost:8554/test"
                - USB摄像头索引，如 0, 1
            buffer_size: OpenCV缓冲区大小，默认1（最低延迟）
        """
        self.source = source
        if isinstance(source, str) and "://" in source:
            parts = urlsplit(source)
            self._source_label = urlunsplit((parts.scheme, parts.hostname or "camera", parts.path, "", ""))
        else:
            self._source_label = "video_file" if isinstance(source, str) else f"camera-{source}"
        self.buffer_size = buffer_size

        # 状态
        self._status = StreamStatus.DISCONNECTED
        self._status_lock = threading.Lock()

        # 帧数据
        self._frame: Optional[cv2.typing.MatLike] = None
        self._frame_lock = threading.Lock()
        self._frame_time: float = 0  # 帧时间戳
        self._generation = uuid.uuid4().hex
        self._source_sequence = 0
        self._frame_stamp = None
        self._on_frame_packet = None

        # 流信息
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0

        # 线程控制
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # 统计信息
        self._frame_count: int = 0
        self._start_time: float = 0
        self._actual_fps: float = 0

        # 回调函数
        self._on_status_change: Optional[Callable[[StreamStatus], None]] = None
        self._on_frame: Optional[Callable[[cv2.typing.MatLike], None]] = None

    @property
    def status(self) -> StreamStatus:
        """获取当前状态"""
        with self._status_lock:
            return self._status

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self.status == StreamStatus.CONNECTED

    @property
    def width(self) -> int:
        """视频宽度"""
        return self._width

    @property
    def height(self) -> int:
        """视频高度"""
        return self._height

    @property
    def fps(self) -> float:
        """视频帧率"""
        return self._fps

    @property
    def actual_fps(self) -> float:
        """实际读取帧率"""
        return self._actual_fps

    @property
    def frame_count(self) -> int:
        """已读取帧数"""
        return self._frame_count

    def set_on_status_change(self, callback: Callable[[StreamStatus], None]):
        """设置状态变化回调"""
        self._on_status_change = callback

    def set_on_frame(self, callback: Callable[[cv2.typing.MatLike], None]):
        """设置新帧回调"""
        self._on_frame = callback

    def set_on_frame_packet(self, callback):
        """Observe original frames and immutable metadata on the reader thread."""
        self._on_frame_packet = callback

    def _set_status(self, status: StreamStatus):
        """设置状态并触发回调"""
        with self._status_lock:
            if self._status != status:
                self._status = status
                if self._on_status_change:
                    try:
                        self._on_status_change(status)
                    except Exception:
                        pass

    def _connect(self) -> bool:
        """
        连接到流

        Returns:
            是否连接成功
        """
        self._set_status(StreamStatus.CONNECTING)

        try:
            # 创建VideoCapture
            if isinstance(self.source, str):
                # RTSP流
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                # USB摄像头
                self._cap = cv2.VideoCapture(self.source)

            # 设置缓冲区大小（减少延迟）
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # 等待连接（最多10秒）
            start = time.time()
            while not self._cap.isOpened():
                if time.time() - start > 10:
                    self._set_status(StreamStatus.ERROR)
                    return False
                time.sleep(0.1)

            # 获取流信息
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)

            self._set_status(StreamStatus.CONNECTED)
            return True

        except Exception as e:
            print(f"[StreamReader] Connection failed ({type(e).__name__}): {self._source_label}")
            self._set_status(StreamStatus.ERROR)
            return False

    def _read_loop(self):
        """读帧循环（在独立线程中运行）"""
        self._start_time = time.time()
        self._frame_count = 0
        fps_update_time = time.time()
        fps_frame_count = 0

        # 判断是否是视频文件（需要帧率控制）
        is_video_file = isinstance(self.source, str) and '://' not in self.source
        frame_interval = 1.0 / self._fps if is_video_file and self._fps > 0 else 0
        last_frame_time = time.time()

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                # 尝试重连
                self._set_status(StreamStatus.RECONNECTING)
                time.sleep(1)
                if not self._connect():
                    continue
                self._generation = uuid.uuid4().hex
                self._source_sequence = 0

            # 视频文件帧率控制
            if is_video_file and frame_interval > 0:
                elapsed = time.time() - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()

            ret, frame = self._cap.read()

            if ret:
                received_at = time.time()
                self._source_sequence += 1
                position = self._cap.get(cv2.CAP_PROP_POS_MSEC) if is_video_file else None
                stamp = FrameStamp(self._source_sequence, self._generation, int(received_at * 1000),
                                   int(received_at * 1000), "receiver_clock", round(position) if position is not None else None)
                # 更新帧
                with self._frame_lock:
                    self._frame = frame
                    self._frame_time = received_at
                    self._frame_stamp = stamp
                if self._on_frame_packet:
                    try:
                        self._on_frame_packet(frame, stamp)
                    except Exception as error:
                        print(f"Frame evidence callback failed ({type(error).__name__})")

                self._frame_count += 1
                fps_frame_count += 1

                # 触发回调
                if self._on_frame:
                    try:
                        self._on_frame(frame)
                    except Exception:
                        pass

                # 每秒更新一次实际帧率
                elapsed = time.time() - fps_update_time
                if elapsed >= 1.0:
                    self._actual_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_update_time = time.time()
            else:
                # 读取失败，可能断线
                if self._running:
                    self._set_status(StreamStatus.RECONNECTING)
                    self._cap.release()
                    self._cap = None
                    time.sleep(0.1)

        # 清理
        if self._cap:
            self._cap.release()
            self._cap = None
        self._set_status(StreamStatus.DISCONNECTED)

    def start(self) -> bool:
        """
        启动流读取

        Returns:
            是否启动成功
        """
        if self._running:
            return True

        # 先尝试连接
        if not self._connect():
            return False

        # 启动读帧线程
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        return True

    def stop(self):
        """停止流读取"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def get_frame(self) -> Optional[cv2.typing.MatLike]:
        """
        获取最新帧

        Returns:
            最新帧，如果没有帧则返回None
        """
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_frame_with_time(self) -> Tuple[Optional[cv2.typing.MatLike], float]:
        """
        获取最新帧及其时间戳

        Returns:
            (帧, 时间戳) 元组
        """
        with self._frame_lock:
            frame = self._frame.copy() if self._frame is not None else None
            return frame, self._frame_time

    def get_frame_after(self, timestamp: float) -> Tuple[Optional[cv2.typing.MatLike], float]:
        """Return a copied frame only when a newer frame is available."""
        with self._frame_lock:
            if self._frame is None or self._frame_time <= timestamp:
                return None, self._frame_time
            return self._frame.copy(), self._frame_time

    def get_frame_packet_after(self, sequence: int, generation: str):
        """Read frame and provenance atomically, including a reconnect epoch."""
        with self._frame_lock:
            stamp = self._frame_stamp
            if self._frame is None or stamp is None or (stamp.generation == generation and stamp.sequence <= sequence):
                return None, stamp
            return self._frame.copy(), stamp

    def get_info(self) -> dict:
        """
        获取流信息

        Returns:
            包含流信息的字典
        """
        return {
            "source": self.source,
            "status": self.status.value,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "actual_fps": self._actual_fps,
            "frame_count": self._frame_count,
        }


# 测试代码
if __name__ == "__main__":
    import sys

    # 默认使用模拟RTSP流
    source = "rtsp://localhost:8554/test"
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # 如果是数字，转为int（USB摄像头）
        if source.isdigit():
            source = int(source)

    print("=" * 50)
    print("  StreamReader 测试")
    print("=" * 50)
    print()
    reader = StreamReader(source)
    print(f"[信息] 流源: {reader._source_label}")
    print()

    # 状态变化回调
    def on_status_change(status):
        print(f"[状态] {status.value}")

    # 创建读取器
    reader.set_on_status_change(on_status_change)

    print("[操作] 正在连接...")
    if not reader.start():
        print("[错误] 连接失败")
        sys.exit(1)

    info = reader.get_info()
    print(f"[信息] 分辨率: {info['width']}x{info['height']}")
    print(f"[信息] 帧率: {info['fps']:.1f} FPS")
    print()
    print("[操作] 显示画面，按 Q 退出")
    print()

    cv2.namedWindow("StreamReader Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("StreamReader Test", 960, 540)

    while True:
        frame = reader.get_frame()

        if frame is not None:
            # 显示信息
            info = reader.get_info()
            cv2.putText(frame, f"Status: {info['status']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {info['actual_fps']:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {info['frame_count']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to exit", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("StreamReader Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    reader.stop()
    cv2.destroyAllWindows()

    print()
    print("[完成] 测试结束")
    print(f"[统计] 共读取 {reader.frame_count} 帧")
