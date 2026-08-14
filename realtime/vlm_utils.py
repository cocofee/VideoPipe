# -*- coding: utf-8 -*-
"""
VLM (Vision Language Model) 辅助工具类
包含提示词模板和结果解析逻辑
"""

import re
import cv2
import base64
import time
import threading
import difflib
import logging
import requests
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 设置日志
logger = logging.getLogger("VideoPipe.VLM")

# ============================================================================
# 异常定义
# ============================================================================

class VLMRateLimitError(Exception):
    """VLM API 速率限制或配额异常"""
    def __init__(self, message, status_code=429):
        super().__init__(message)
        self.status_code = status_code
        self.is_quota_exceeded = "quota" in message.lower() or "insufficient_balance" in message.lower()

# ============================================================================
# VLM 提示词模板
# ============================================================================

PROMPT_BACKGROUND_CHECK = """你是一个体育赛事图像分析助手。请判断图中主体的类别。

【类别定义】
- ATHLETE: 正在运动中、准备通过终点的参赛选手（跑步、骑行）。特征：身体前倾、发力感、戴头盔/号码布、有明显的行进动态。
- GATE: 赛道设施。特征：横跨赛道、桁架结构、固定不动、挂有计时设备或横幅。
- BYSTANDER: 非参赛人员。特征：站立不动、坐着、背对赛道、无号码布、手持相机、姿态放松。
- OTHER: 无法判断或其他物体（如车辆、流浪小动物、被风吹动的旗帜）。

【判定要点】
1. 动态 vs 静态：运动员必须有明显的行进趋势。
2. 装备：运动员通常佩戴头盔、紧身运动服或挂有号码布。
3. 位置：运动员通常在赛道正中间，而观众在两侧。

【输出格式】
第一行：类别（只写一个词：ATHLETE/GATE/BYSTANDER/OTHER）
第二行：置信度（0.0-1.0之间的小数）
第三行：简短理由（15字以内）"""

PROMPT_COUNT_ATHLETES = """你是一个体育赛事计数助手。请数一下图中【正在通过终点/赛道的运动员】有几人。

【计数规则】
1. 只数正在运动中的参赛选手（穿号码布的）
2. 不要数观众、工作人员、摄影师
3. 如果有人被遮挡但能看到部分身体，也算1人
4. 如果完全看不到人，回答0

【输出格式】
第一行：数字（0/1/2/3/4/5...）
第二行：简短说明每个人的位置

示例输出：
3
左1人穿红衣，中间2人并排"""

PROMPT_PERSON_MATCH = """你是体育赛事视频中的运动员身份比对助手。请判断两张图片中的运动员是否为同一个人。

输出严格三行：
第一行：SAME、DIFFERENT 或 UNCERTAIN
第二行：置信度（0.0-1.0）
第三行：简短理由"""

PROMPT_BIB_OCR_ENHANCED = """你是专业的体育比赛号码识别专家。请仔细识别图中运动员身上的参赛号码。

【本场比赛号码特征】
- 格式规则: 【1-6位数字】，或【1-3个大写字母前缀 + 1-6位数字】
- 示例: 5, 23, 1056, A123, AB7
- 外观特征: 白底黑字或深色底白字，粗体数字，字高通常15-30像素
- 位置特征: 背部、前胸、头盔、大腿或车座下方，可能部分被身体或器材遮挡

【严格识别规则】
1. 字符限制: 只允许识别【大写字母A-Z】和【数字0-9】
2. 严禁识别: "安宁"、"马拉松"、"经校"、"2024"等赛事名称汉字
3. 严禁识别: 赞助商名称、广告文字
4. 如果只看到部分号码(如"A12__"，后两位模糊)，可以推断但必须标注为低置信度
5. 如果完全无法识别，输出 UNREADABLE

【常见错误校正】
- 字母O和数字0易混淆: 如果是开头字母，通常是O；如果在数字中，通常是0
- 数字1和字母I易混淆: 号码中通常是数字1
- 数字5和字母S易混淆: 号码中通常是数字5
- 数字8和字母B易混淆: 号码中通常是数字8

【输出格式】(严格按照以下3行输出)
第一行: 号码 (如 A123、23 或 UNREADABLE)
第二行: 置信度 (0.0-1.0之间的小数)
第三行: 说明 (10字内，如"清晰可见"或"部分遮挡推断")"""

def build_ocr_prompt_with_list_enhanced(athlete_list: List[str], only_numeric: bool = False) -> str:
    """构建带选手名单的增强版OCR提示词"""
    if not athlete_list:
        return PROMPT_BIB_OCR_ENHANCED

    # 取前100名选手作为参考
    sample_bibs = athlete_list[:100]
    
    # 格式化名单显示
    bib_str = ""
    for i in range(0, len(sample_bibs), 10):
        bib_str += ", ".join(sample_bibs[i:i+10]) + "\n"

    rule_text = "只允许识别【数字0-9】" if only_numeric else "只允许识别【大写字母A-Z】和【数字0-9】"
    format_text = "1-6位纯数字" if only_numeric else "1-6位数字，或1-3个大写字母前缀加1-6位数字"
    example = "2350" if only_numeric else "A123"

    return f"""你是专业的体育比赛号码识别专家。请仔细识别图中运动员身上的参赛号码。

【本场比赛参考名单】(显示前{len(sample_bibs)}名)
{bib_str}

【识别优先级】
1. 优先匹配上述名单中的号码
2. 如果看到的号码与名单中某个号码很接近，倾向于匹配名单号码

【号码特征】
- 格式: {format_text}
- 外观: 白底黑字或深色底白字，粗体数字
- 位置: 背部腰际、前胸或车架横杆

【严格规则】
1. {rule_text}
2. 严禁识别赛事名称和赞助商广告汉字
3. 如果部分模糊，可推断但标注低置信度

【输出格式】(严格3行)
第一行: 号码 (如 {example})
第二行: 置信度 (0.0-1.0)
第三行: 在名单中? (YES/NO/SIMILAR) + 简短说明"""

# ============================================================================
# 结果解析逻辑
# ============================================================================

def parse_background_result(response: str) -> Tuple[str, float, str]:
    """解析 VLM 背景判定结果"""
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    
    category = "OTHER"
    confidence = 0.5
    reason = ""
    
    if len(lines) >= 1:
        cat = lines[0].upper()
        for valid in ["ATHLETE", "GATE", "BYSTANDER", "OTHER"]:
            if valid in cat:
                category = valid
                break
    
    if len(lines) >= 2:
        match = re.search(r'\d+\.?\d*', lines[1])
        if match:
            confidence = float(match.group())
            if confidence > 1.0: confidence /= 100.0
            confidence = max(0.0, min(1.0, confidence))
    
    if len(lines) >= 3:
        reason = lines[2][:20]
    
    return category, confidence, reason

def parse_count_result(response: str) -> Tuple[int, str]:
    """解析人数计数结果"""
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    count = 1
    description = ""
    if lines:
        match = re.search(r'\d+', lines[0])
        count = int(match.group()) if match else 1
    if len(lines) >= 2:
        description = lines[1]
    return count, description

def parse_ocr_result(response: str) -> Tuple[Optional[str], float, str]:
    """解析 VLM OCR 结果"""
    cleaned_response = re.sub(
        r"<think>.*?</think>",
        "",
        str(response or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    lines = [l.strip() for l in cleaned_response.strip().split('\n') if l.strip()]
    if not lines:
        return None, 0.0, "Empty response"
        
    raw = lines[0].strip().strip("`*_ ").upper()
    if any(word in raw for word in ["UNREADABLE", "NO_BIB", "无法识别", "NONE"]):
        return None, 0.0, raw
        
    # 提取号码：大写字母(可选) + 2-6位数字
    bib_number = None
    strict_match = re.fullmatch(r"(?:[A-Z]{1,3})?\d{1,6}", raw)
    if strict_match:
        bib_number = strict_match.group()
    else:
        labeled_match = re.fullmatch(
            r"(?:BIB(?:\s+NUMBER)?|NUMBER)\s*[:#-]\s*([A-Z]{0,3}\d{1,6})",
            raw,
        )
        if labeled_match:
            bib_number = labeled_match.group(1)
    
    confidence = 0.0
    if len(lines) >= 2:
        match = re.search(r'\d+\.?\d*', lines[1])
        if match:
            confidence = float(match.group())
            if confidence > 1.0: confidence /= 100.0
            confidence = max(0.0, min(1.0, confidence))
            
    note = lines[2] if len(lines) >= 3 else ""
    return bib_number, confidence, note

# ============================================================================
# VLM 基础调用类
# ============================================================================

class DoubaoVLMAssistant:
    """豆包 VLM 辅助识别"""
    def __init__(self, api_key: str, endpoint_id: str = None):
        self.api_key = api_key
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/responses"
        self.model = endpoint_id or "doubao-seed-1-8-251228"
    
    def call(self, images: Union[np.ndarray, List[np.ndarray]], prompt: str) -> str:
        """调用 API"""
        if isinstance(images, np.ndarray): images = [images]
        
        content = []
        for img in images:
            if img is None or img.size == 0: continue
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"})
            
        if not content: return ""
        content.append({"type": "input_text", "text": prompt})
        
        try:
            payload = {"model": self.model, "input": [{"role": "user", "content": content}]}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            elif resp.status_code == 429:
                raise VLMRateLimitError(f"Doubao API Rate Limit: {resp.text}", status_code=429)
            return f"Error: {resp.status_code}"
        except VLMRateLimitError:
            raise
        except Exception as e:
            return f"Exception: {str(e)}"

    def check_background(self, crop: np.ndarray) -> Tuple[bool, float, str]:
        """判断是否为运动员（非背景）"""
        result = self.call(crop, PROMPT_BACKGROUND_CHECK)
        category, conf, reason = parse_background_result(result)
        is_athlete = (category == "ATHLETE")
        return is_athlete, conf, reason
    
    def count_athletes(self, region: np.ndarray) -> Tuple[int, str]:
        """计数运动员"""
        result = self.call(region, PROMPT_COUNT_ATHLETES)
        return parse_count_result(result)
    
    def ocr_bib(self, bib_crop: np.ndarray, athlete_list: List[str] = None,
                only_numeric: bool = False, prompt: str = None) -> Tuple[Optional[str], float, str]:
        """OCR 号码牌"""
        if not prompt:
            if athlete_list:
                prompt = build_ocr_prompt_with_list_enhanced(athlete_list, only_numeric)
            else:
                prompt = PROMPT_BIB_OCR_ENHANCED

        result = self.call(bib_crop, prompt)
        return parse_ocr_result(result)


class QwenVLMAssistant:
    """通义千问 (Qwen-VL) 辅助识别"""
    
    def __init__(self, api_key: str, model: str = "qwen-vl-max"):
        self.api_key = api_key
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-conversation/generation"
        self.model = model
        self.timeout = 30 # 增加超时时间到 30s
    
    def _call_api(self, images: Union[np.ndarray, List[np.ndarray]], prompt: str, max_tokens: int = 100) -> str:
        """调用 DashScope API (支持单图或多图)"""
        if isinstance(images, np.ndarray):
            images = [images]
            
        content = []
        for img in images:
            if img is None or img.size == 0: continue
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            content.append({"image": f"data:image/jpeg;base64,{img_base64}"})
            
        if not content:
            return ""
            
        content.append({"text": prompt})
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": {
                        "messages": [{
                            "role": "user",
                            "content": content
                        }]
                    },
                    "parameters": {
                        "max_tokens": max_tokens,
                        "temperature": 0.1
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                res_data = response.json()
                try:
                    if 'output' in res_data and 'choices' in res_data['output']:
                        choice = res_data['output']['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            if isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
                                return content[0]['text']
                            elif isinstance(content, str):
                                return content
                    return str(res_data)
                except Exception as e:
                    logger.warning(f"[VLM-Qwen] 结果解析异常: {e}")
                    return str(res_data)
            elif response.status_code == 429:
                raise VLMRateLimitError(f"Qwen API Rate Limit: {response.text}", status_code=429)
            else:
                logger.error(f"[VLM-Qwen] API 错误: {response.status_code} {response.text}")
                return ""
                
        except VLMRateLimitError:
            raise
        except Exception as e:
            logger.error(f"[VLM-Qwen] 请求异常: {e}")
            return ""

    def check_background(self, crop: np.ndarray) -> Tuple[bool, float, str]:
        """判断是否为运动员（非背景）"""
        result = self._call_api(crop, PROMPT_BACKGROUND_CHECK, max_tokens=50)
        category, conf, reason = parse_background_result(result)
        is_athlete = (category == "ATHLETE")
        return is_athlete, conf, reason
    
    def count_athletes(self, region: np.ndarray) -> Tuple[int, str]:
        """计数运动员"""
        result = self._call_api(region, PROMPT_COUNT_ATHLETES, max_tokens=80)
        return parse_count_result(result)
    
    def ocr_bib(self, bib_crop: np.ndarray, athlete_list: List[str] = None,
                only_numeric: bool = False, prompt: str = None) -> Tuple[Optional[str], float, str]:
        """OCR 号码牌"""
        if not prompt:
            if athlete_list:
                prompt = build_ocr_prompt_with_list_enhanced(athlete_list, only_numeric)
            else:
                prompt = PROMPT_BIB_OCR_ENHANCED

        result = self._call_api(bib_crop, prompt, max_tokens=120)
        return parse_ocr_result(result)


class OpenAIVLMAssistant:
    """OpenAI-compatible vision assistant for direct or proxy endpoints."""

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", base_url: str = None):
        self.api_key = api_key
        self.model = model or "gpt-4.1-mini"
        self.base_url = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
        self.api_url = self._resolve_chat_url(self.base_url)
        self.timeout = 30

    @staticmethod
    def _resolve_chat_url(base_url: str) -> str:
        """Accept either a /v1 base URL or a full chat/completions URL."""
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    def call(self, images: Union[np.ndarray, List[np.ndarray]], prompt: str,
             max_tokens: int = 120) -> str:
        """Send one or more images and return the model's text response."""
        if isinstance(images, np.ndarray):
            images = [images]

        content = []
        for image in images:
            if image is None or image.size == 0:
                continue
            ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
            })

        if not content:
            return ""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": item["image_url"]}}
                    for item in content
                    if item["type"] == "input_image"
                ],
            ]}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 429:
                raise VLMRateLimitError(
                    "OpenAI API rate limit (HTTP 429)",
                    status_code=429,
                )
            if response.status_code != 200:
                logger.error("[VLM-OpenAI] API error: status=%s", response.status_code)
                return ""

            data = response.json()
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {}) or {}
                message_content = message.get("content", "")
                if isinstance(message_content, str) and message_content.strip():
                    return message_content
                if isinstance(message_content, list):
                    visible_text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in message_content
                    )
                    if visible_text.strip():
                        return visible_text
            # Some OpenAI-compatible relays return the visible answer in a
            # reasoning field when the selected model emits a thinking block.
            # Preserve that text as a last-resort parser input instead of
            # treating a successful HTTP response as an empty OCR result.
            if choices:
                message = choices[0].get("message", {}) or {}
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                if isinstance(reasoning, str):
                    return reasoning
            return ""
        except VLMRateLimitError:
            raise
        except Exception as exc:
            logger.error("[VLM-OpenAI] request failed: %s", exc)
            return ""

    def check_background(self, crop: np.ndarray) -> Tuple[bool, float, str]:
        result = self.call(crop, PROMPT_BACKGROUND_CHECK, max_tokens=50)
        category, confidence, reason = parse_background_result(result)
        return category == "ATHLETE", confidence, reason

    def count_athletes(self, region: np.ndarray) -> Tuple[int, str]:
        result = self.call(region, PROMPT_COUNT_ATHLETES, max_tokens=80)
        return parse_count_result(result)

    def ocr_bib(self, bib_crop: np.ndarray, athlete_list: List[str] = None,
                only_numeric: bool = False, prompt: str = None) -> Tuple[Optional[str], float, str]:
        if not prompt:
            prompt = (
                build_ocr_prompt_with_list_enhanced(athlete_list, only_numeric)
                if athlete_list else PROMPT_BIB_OCR_ENHANCED
            )
        result = self.call(bib_crop, prompt, max_tokens=120)
        return parse_ocr_result(result)

    def compare_persons(self, crops: List[np.ndarray]) -> Tuple[str, float, str]:
        result = self.call(crops, PROMPT_PERSON_MATCH, max_tokens=80)
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        conclusion = "UNCERTAIN"
        if lines:
            raw = lines[0].upper()
            if "SAME" in raw:
                conclusion = "SAME"
            elif "DIFFERENT" in raw:
                conclusion = "DIFFERENT"
        confidence = 0.0
        if len(lines) >= 2:
            match = re.search(r"\d+\.?\d*", lines[1])
            if match:
                confidence = float(match.group())
                if confidence > 1.0:
                    confidence /= 100.0
                confidence = max(0.0, min(1.0, confidence))
        reason = lines[2] if len(lines) >= 3 else ""
        return conclusion, confidence, reason


class AsyncVLMPipeline:
    """异步 VLM 处理管道"""
    
    def __init__(self, vlm_assistant: Any, max_workers: int = 4):
        self.vlm = vlm_assistant
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 结果缓存
        self.background_cache: Set[int] = set()
        self.athlete_cache: Set[int] = set()
        self.ocr_cache: Dict[int, Tuple[str, float]] = {}
        self.match_cache: Dict[str, Tuple[str, float, str]] = {}  # key: "tid1-tid2"
        self.pending_ocr: Set[int] = set() # 正在进行的 OCR 任务
        
        # 速率限制
        self.call_times = deque(maxlen=60)
        self.max_calls_per_minute = 30
        self.cooldown_until = 0.0
        self.is_quota_exhausted = False
        
        self._lock = threading.Lock()
    
    def _can_call(self, is_priority: bool = False) -> bool:
        """检查是否可以调用 API"""
        now = time.time()
        with self._lock:
            # 1. 检查冷却时间
            if now < self.cooldown_until:
                return False
            
            # 2. 如果配额耗尽，彻底停止调用
            if self.is_quota_exhausted:
                return False
                
            # 3. 速率限制检查
            while self.call_times and now - self.call_times[0] > 60:
                self.call_times.popleft()
            
            # 非优先级任务限制更严
            limit = self.max_calls_per_minute if is_priority else (self.max_calls_per_minute // 2)
            return len(self.call_times) < limit
    
    def _handle_error(self, e: Exception):
        """处理 VLM 错误并设置冷却"""
        now = time.time()
        with self._lock:
            if isinstance(e, VLMRateLimitError):
                if e.is_quota_exceeded:
                    self.is_quota_exhausted = True
                    self.cooldown_until = now + 300 # 5分钟冷却
                    logger.error("="*50)
                    logger.error("VLM 配额已耗尽！系统将暂停 VLM 任务 5 分钟。")
                    logger.error("="*50)
                else:
                    self.cooldown_until = now + 30 # 30秒冷却
                    logger.warning(f"[VLM] 触发频率限制，进入 30s 冷却期。")
            else:
                # 其他异常，小额冷却
                self.cooldown_until = now + 5
    
    def _record_call(self):
        """记录 API 调用"""
        with self._lock:
            self.call_times.append(time.time())
    
    def submit_background_check(self, track_id: int, crop: np.ndarray):
        """提交背景判定任务 (低优先级)"""
        if track_id in self.background_cache or track_id in self.athlete_cache:
            return
        if not self._can_call(is_priority=False):
            return
        
        self._record_call()
        future = self.executor.submit(self.vlm.check_background, crop)
        future.add_done_callback(lambda f: self._on_background_result(track_id, f))
    
    def _on_background_result(self, track_id: int, future):
        """背景判定结果回调"""
        try:
            is_athlete, conf, reason = future.result()
            with self._lock:
                if is_athlete:
                    self.athlete_cache.add(track_id)
                else:
                    self.background_cache.add(track_id)
                    logger.info(f"[VLM] ID {track_id} 判定为背景: {reason}")
        except Exception as e:
            self._handle_error(e)
            logger.error(f"[VLM] 背景判定失败: {e}")
    
    def submit_ocr(self, track_id: int, bib_crop: np.ndarray, athlete_list: List[str] = None, only_numeric: bool = False):
        """提交 OCR 任务 (高优先级)"""
        with self._lock:
            if track_id in self.ocr_cache or track_id in self.pending_ocr:
                return
        
        if not self._can_call(is_priority=True):
            return
        
        with self._lock:
            self.pending_ocr.add(track_id)
            
        self._record_call()
        future = self.executor.submit(self.vlm.ocr_bib, bib_crop, athlete_list, only_numeric)
        future.add_done_callback(lambda f: self._on_ocr_result(track_id, f))
    
    def is_ocr_pending(self, track_id: int) -> bool:
        """检查 OCR 是否正在进行"""
        with self._lock:
            return track_id in self.pending_ocr
    
    def get_status(self) -> dict:
        """获取当前管道状态"""
        with self._lock:
            now = time.time()
            return {
                'pending_ocr': len(self.pending_ocr),
                'cooldown': max(0, int(self.cooldown_until - now)),
                'quota_exhausted': self.is_quota_exhausted,
                'calls_per_minute': len(self.call_times)
            }
    
    def _on_ocr_result(self, track_id: int, future):
        """OCR 结果回调"""
        try:
            text, conf, note = future.result()
            with self._lock:
                if track_id in self.pending_ocr:
                    self.pending_ocr.remove(track_id)
                if text and conf > 0.3:
                    self.ocr_cache[track_id] = (text, conf)
            
            if text:
                logger.info(f"[VLM-OCR] ID {track_id} -> {text} (conf={conf:.2f}, note={note})")
        except Exception as e:
            self._handle_error(e)
            with self._lock:
                if track_id in self.pending_ocr:
                    self.pending_ocr.remove(track_id)
            logger.error(f"[VLM-OCR] ID {track_id} 失败: {e}")
    
    def is_known_background(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self.background_cache
    
    def get_cached_bib(self, track_id: int) -> Optional[Tuple[str, float]]:
        with self._lock:
            return self.ocr_cache.get(track_id)
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=False)

# ============================================================================
# 图像增强与 OCR 工具函数
# ============================================================================

COMMON_OCR_ERRORS = {
    '0': ['O', 'D', 'Q', 'o'],
    '1': ['I', 'L', 'J', 'i', 'l'],
    '2': ['Z', 'z'],
    '4': ['A'],
    '5': ['S', 's'],
    '6': ['G', 'g'],
    '8': ['B', 'b'],
    'A': ['4'],
    'B': ['8'],
    'S': ['5'],
    'Z': ['2'],
    'O': ['0'],
    'I': ['1'],
    'G': ['6'],
    'D': ['0'],
    'Q': ['0']
}

def fuzzy_match_bib(bib: str, athlete_list: set, threshold: float = 0.7) -> Optional[str]:
    """更智能的号码模糊匹配"""
    if not bib or not athlete_list:
        return None
    
    if bib in athlete_list:
        return bib
        
    for i, char in enumerate(bib):
        if char in COMMON_OCR_ERRORS:
            for replacement in COMMON_OCR_ERRORS[char]:
                candidate = bib[:i] + replacement + bib[i+1:]
                if candidate in athlete_list:
                    return candidate
                    
    matches = difflib.get_close_matches(bib, list(athlete_list), n=1, cutoff=threshold)
    if matches:
        return matches[0]
        
    return None

# ============================================================================
# 几何计算与图像工具函数
# ============================================================================

def point_side_of_line(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
    """计算点相对于线的位置"""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def point_to_line_dist(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
    """计算点到直线的垂直距离"""
    dx = x2 - x1
    dy = y2 - y1
    line_len = np.sqrt(dx * dx + dy * dy)
    if line_len == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    return abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / line_len

def bbox_iou(a: List[int], b: List[int]) -> float:
    """计算两个边界框的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aw, ah = max(0, ax2 - ax1), max(0, ay2 - ay1)
    bw, bh = max(0, bx2 - bx1), max(0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return (inter / union) if union > 0 else 0.0

def read_image_unicode(path: Path) -> Optional[np.ndarray]:
    """支持 Unicode 路径的图像读取"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def write_image_unicode(path: Path, image: np.ndarray) -> bool:
    """支持 Unicode 路径的图像写入"""
    try:
        ext = path.suffix if path.suffix else ".jpg"
        success, encoded = cv2.imencode(ext, image)
        if not success:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False
