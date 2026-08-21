"""Read-only Hikvision camera clock verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Callable, Optional
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import requests
from requests.auth import HTTPDigestAuth


DEFAULT_MAX_SKEW_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class CameraClockCheck:
    ok: bool
    message: str
    skew_ms: Optional[int] = None
    camera_time: str = ""
    time_mode: str = ""


def _element_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == name:
            return str(element.text or "").strip()
    return ""


def parse_hikvision_time(payload: bytes | str) -> tuple[datetime, str]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8-sig")
    root = ET.fromstring(payload)
    local_time = _element_text(root, "localTime")
    if not local_time:
        raise ValueError("camera response does not contain localTime")
    parsed = datetime.fromisoformat(local_time.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("camera localTime does not include a timezone offset")
    return parsed, _element_text(root, "timeMode")


def rtsp_camera_credentials(source: object) -> Optional[tuple[str, str, str]]:
    if not isinstance(source, str) or not source.lower().startswith("rtsp://"):
        return None
    try:
        parsed = urlsplit(source)
    except ValueError:
        return None
    if not parsed.hostname or parsed.username is None or parsed.password is None:
        return None
    return (
        parsed.hostname,
        unquote(parsed.username),
        unquote(parsed.password),
    )


def camera_clock_check_required(source: object) -> bool:
    """Return whether an RTSP source carries credentials for device checks."""
    return rtsp_camera_credentials(source) is not None


def _management_time_url(source: object, management_url: Optional[str]) -> str:
    credentials = rtsp_camera_credentials(source)
    if credentials is None:
        raise ValueError("RTSP 地址缺少相机登录凭据")

    host = credentials[0]
    raw_url = str(management_url or "").strip()
    if not raw_url:
        host_for_url = f"[{host}]" if ":" in host else host
        return f"http://{host_for_url}/ISAPI/System/time"

    parsed = urlsplit(raw_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("管理地址必须使用 HTTP 或 HTTPS")
    if parsed.hostname.lower() != host.lower():
        raise ValueError("管理地址主机必须与 RTSP 相机主机一致")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("管理地址不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise ValueError("管理地址不能包含查询参数或片段")
    if parsed.path not in {"", "/"}:
        raise ValueError("管理地址只能包含协议、主机和端口")

    host_for_url = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = parsed.port
    port_text = f":{port}" if port is not None else ""
    return (
        f"{parsed.scheme.lower()}://{host_for_url}{port_text}"
        "/ISAPI/System/time"
    )


def check_hikvision_camera_clock(
    source: object,
    *,
    management_url: Optional[str] = None,
    timeout: float = 3.0,
    max_skew_seconds: float = DEFAULT_MAX_SKEW_SECONDS,
    request_get: Optional[Callable[..., object]] = None,
    clock: Optional[Callable[[], float]] = None,
) -> CameraClockCheck:
    credentials = rtsp_camera_credentials(source)
    if credentials is None:
        return CameraClockCheck(False, "缺少海康管理地址或登录凭据，无法核验相机时间")

    _host, username, password = credentials
    get = requests.get if request_get is None else request_get
    now = time.time if clock is None else clock

    try:
        url = _management_time_url(source, management_url)
        started_at = float(now())
        response = get(
            url,
            auth=HTTPDigestAuth(username, password),
            timeout=float(timeout),
        )
        finished_at = float(now())
        status_code = int(getattr(response, "status_code", 0))
        if status_code != 200:
            return CameraClockCheck(False, f"无法读取相机时间：HTTP {status_code}")
        payload = getattr(response, "content", b"")
        if not payload:
            payload = str(getattr(response, "text", ""))
        camera_time, time_mode = parse_hikvision_time(payload)
    except ValueError as error:
        return CameraClockCheck(False, f"相机管理地址无效：{error}")
    except Exception as error:
        return CameraClockCheck(
            False,
            f"无法读取相机时间（{type(error).__name__}）",
        )

    reference_time = (started_at + finished_at) / 2.0
    skew_ms = int(round((camera_time.timestamp() - reference_time) * 1000.0))
    camera_text = camera_time.isoformat(sep=" ", timespec="seconds")
    mode_text = time_mode or "未知模式"
    if abs(skew_ms) <= int(round(float(max_skew_seconds) * 1000.0)):
        return CameraClockCheck(
            True,
            f"相机时间已核验：{camera_text}，与电脑相差 {skew_ms:+d} ms（{mode_text}）",
            skew_ms=skew_ms,
            camera_time=camera_text,
            time_mode=time_mode,
        )
    return CameraClockCheck(
        False,
        f"相机时间异常：{camera_text}，与电脑相差 {skew_ms:+d} ms；请先校时（{mode_text}）",
        skew_ms=skew_ms,
        camera_time=camera_text,
        time_mode=time_mode,
    )


__all__ = [
    "CameraClockCheck",
    "DEFAULT_MAX_SKEW_SECONDS",
    "camera_clock_check_required",
    "check_hikvision_camera_clock",
    "parse_hikvision_time",
    "rtsp_camera_credentials",
]
