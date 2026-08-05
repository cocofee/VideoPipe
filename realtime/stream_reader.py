"""
流读取模块

支持RTSP摄像头和USB摄像头的实时流读取
使用独立线程持续读帧，不阻塞主程序
"""

import cv2
import threading
import time
from typing import Optional, Tuple, Callable
from enum import Enum

try:
    from .logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("StreamReader")


class StreamStatus(Enum):
    """流状态枚举"""
    DISCONNECTED = "disconnected"  # 未连接
    CONNECTING = "connecting"      # 连接中
    CONNECTED = "connected"        # 已连接
    RECONNECTING = "reconnecting"  # 重连中
    ENDED = "ended"                # 本地视频已播放完成
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
        self.buffer_size = buffer_size

        # 状态
        self._status = StreamStatus.DISCONNECTED
        self._status_lock = threading.Lock()

        # 帧数据
        self._frame: Optional[cv2.typing.MatLike] = None
        self._frame_lock = threading.Lock()
        self._frame_time: float = 0  # 帧时间戳

        # 流信息
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0

        # 线程控制
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._eof_reached = False

        # 统计信息
        self._frame_count: int = 0
        self._start_time: float = 0
        self._actual_fps: float = 0
        self._last_frame_time: float = 0  # 上一帧接收的时间戳

        # 重连参数
        self._max_retry_interval = 5.0
        self._watchdog_timeout = 3.0  # 3秒没新帧认为断线
        self._initial_retry_interval = 1.0  # 首次重连等待1秒

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

    @staticmethod
    def is_video_file_source(source) -> bool:
        """Return whether the source is a finite local video input."""
        return isinstance(source, str) and not source.lower().startswith("rtsp")

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
            if isinstance(self.source, str) and self.source.startswith('rtsp'):
                # RTSP流 - 使用 FFMPEG 插件
                # 修复：通过环境变量设置超时，避免连接时卡死
                import os
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp;timeout;5000000'

                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                # 设置读取超时（连接后的读取超时）
                self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            elif isinstance(self.source, int):
                # USB摄像头 - 在 Windows 上优先使用 DSHOW 以支持高帧率 (如 Action 5)
                # DSHOW 启动比默认的 MSMF 快且稳定
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(self.source)

                # 优化 USB 摄像头性能
                if self._cap.isOpened():
                    # 1. 优先请求 MJPG 格式以获得更高帧率和更低延迟
                    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    # 2. 请求 60 FPS (Action 5 支持)
                    self._cap.set(cv2.CAP_PROP_FPS, 60)
                    # 3. 设置常用分辨率
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                # 模拟视频文件或其他
                self._cap = cv2.VideoCapture(self.source)

            # 修复：检查是否成功打开，避免无限等待
            if not self._cap.isOpened():
                logger.error(f"[StreamReader] 无法打开视频源 {self.source}")
                self._set_status(StreamStatus.ERROR)
                return False

            # 设置缓冲区大小（0或1代表最低延迟，Action 5 建议 1）
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 尝试读取第一帧，确保流真的可用（最多5秒）
            start = time.time()
            first_frame_ok = False
            while time.time() - start < 5:
                ret, _ = self._cap.read()
                if ret:
                    first_frame_ok = True
                    break
                time.sleep(0.1)

            if not first_frame_ok:
                logger.error(f"[StreamReader] 无法从视频源读取帧: {self.source}")
                self._cap.release()
                self._cap = None
                self._set_status(StreamStatus.ERROR)
                return False

            if self.is_video_file_source(self.source):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 获取实际的流信息
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"[StreamReader] 连接成功: {self.source} ({self._width}x{self._height} @ {self._fps}fps)")

            # 获取实际格式
            fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

            logger.info(f"[StreamReader] 相机配置: {self._width}x{self._height} @ {self._fps:.1f}fps ({fourcc_str})")

            self._set_status(StreamStatus.CONNECTED)
            return True

        except Exception as e:
            logger.exception(f"[StreamReader] 连接发生异常: {e}")
            self._set_status(StreamStatus.ERROR)
            return False

    def _read_loop(self):
        """读帧循环（在独立线程中运行）"""
        self._start_time = time.time()
        self._frame_count = 0
        fps_update_time = time.time()
        fps_frame_count = 0
        self._last_frame_time = time.time()

        # 判断是否是视频文件（需要帧率控制）
        is_video_file = self.is_video_file_source(self.source)
        frame_interval = 1.0 / self._fps if is_video_file and self._fps > 0 else 0
        
        retry_count = 0

        while self._running:
            # 1. 检查连接状态和看门狗
            now = time.time()
            is_stale = (not is_video_file) and (now - self._last_frame_time > self._watchdog_timeout)
            
            if self._cap is None or not self._cap.isOpened() or is_stale:
                if is_stale:
                    logger.warning(f"[StreamReader] 警告: {self._watchdog_timeout}s 没收到新帧，触发重连...")
                
                # 尝试重连
                self._set_status(StreamStatus.RECONNECTING)
                if self._cap:
                    self._cap.release()
                    self._cap = None
                
                logger.info(f"[StreamReader] 正在尝试重新连接视频源: {self.source}...")
                if not self._connect():
                    retry_count += 1
                    # 修复：使用指数退避算法，而不是线性增长
                    # wait_time = 1, 2, 4, 5, 5, 5...（最大5秒）
                    wait_time = min(self._max_retry_interval,
                                  self._initial_retry_interval * (2 ** (retry_count - 1)))
                    logger.error(f"[StreamReader] 重连失败 ({retry_count}次)，{wait_time:.1f}s 后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    retry_count = 0
                    logger.info(f"[StreamReader] 重连成功！")
                    self._last_frame_time = time.time() # 重置看门狗

            # 2. 视频文件帧率控制
            if is_video_file and frame_interval > 0:
                elapsed = time.time() - self._last_frame_time
                if elapsed < frame_interval:
                    time.sleep(max(0, frame_interval - elapsed))

            # 3. 读取帧
            ret, frame = self._cap.read()

            if ret:
                self._last_frame_time = time.time()
                # 更新帧（使用引用而不是拷贝，避免内存泄漏）
                # OpenCV内部缓冲区复用可以通过缓冲区大小设置来避免
                with self._frame_lock:
                    self._frame = frame
                    self._frame_time = self._last_frame_time

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
                if is_video_file:
                    logger.info(f"[StreamReader] 本地视频播放完成: {self.source}")
                    self._eof_reached = True
                    self._running = False
                    self._set_status(StreamStatus.ENDED)
                    break

                # 实时流读取失败，按断线处理
                if self._running:
                    self._set_status(StreamStatus.RECONNECTING)
                    # 释放旧连接，强制下次循环重新连接
                    if self._cap:
                        self._cap.release()
                        self._cap = None
                    time.sleep(1.0)  # 等待一秒后重试

        # 清理
        if self._cap:
            self._cap.release()
            self._cap = None
        if not self._eof_reached:
            self._set_status(StreamStatus.DISCONNECTED)

    def start(self) -> bool:
        """
        启动流读取

        Returns:
            是否启动成功
        """
        if self._running:
            return True

        self._eof_reached = False

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
            # 返回拷贝以避免外部修改，但添加内存管理
            return self._frame.copy() if self._frame is not None else None

    def get_frame_with_time(self) -> Tuple[Optional[cv2.typing.MatLike], float]:
        """
        获取最新帧及其时间戳

        Returns:
            (帧, 时间戳) 元组
        """
        with self._frame_lock:
            # 返回拷贝以避免外部修改
            frame = self._frame.copy() if self._frame is not None else None
            return frame, self._frame_time

    def get_frame_after(self, timestamp: float) -> Tuple[Optional[cv2.typing.MatLike], float]:
        """Return a copied frame only when a newer frame is available."""
        with self._frame_lock:
            frame_time = self._frame_time
            if self._frame is None or frame_time <= timestamp:
                return None, frame_time
            return self._frame.copy(), frame_time

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
            "eof_reached": self._eof_reached,
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
    print(f"[信息] 流源: {source}")
    print()

    # 状态变化回调
    def on_status_change(status):
        print(f"[状态] {status.value}")

    # 创建读取器
    reader = StreamReader(source)
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
