import time
import uuid
from typing import Any, Dict


def generate_chat_id(session_id: str) -> str:
    """生成聊天记录ID"""
    timestamp = int(time.time() * 1000)
    return f"{session_id}-{timestamp}"


def generate_uuid() -> str:
    """生成UUID"""
    return str(uuid.uuid4())


def get_current_timestamp() -> int:
    """获取当前时间戳（毫秒）"""
    return int(time.time() * 1000)


def success_response(data: Any = None) -> Dict:
    """成功响应"""
    return {
        "code": 200,
        "message": "success",
        "data": data
    }


def error_response(code: int, message: str) -> Dict:
    """错误响应"""
    return {
        "code": code,
        "message": message
    }
