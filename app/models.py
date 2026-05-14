from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime


class Response(BaseModel):
    """统一响应模型"""
    code: int
    message: str
    data: Optional[Any] = None


class AiChatRecord(BaseModel):
    """聊天记录模型"""
    chat_id: str
    user_id: str = ""
    session_id: str
    ai_model: str
    role: str  # user/assistant
    content: str
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    create_time: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentGetRecordRequest(BaseModel):
    """获取会话记录请求"""
    session_id: str
    cursor: int = 0
    action: str = "normal"
    limit: int = 30


class AgentGetRecordResponse(BaseModel):
    """获取会话记录响应"""
    list: List[AiChatRecord]


class SpeechTTSRequest(BaseModel):
    """TTS请求"""
    text: str
    voice: Optional[str] = "default"
    speed: Optional[int] = 5


class SpeechTTSResponse(BaseModel):
    """TTS响应"""
    audio_url: str


class SpeechASRRequest(BaseModel):
    """ASR请求"""
    encode: str


class SpeechASRResponse(BaseModel):
    """ASR响应"""
    text: str


class WebSocketContentItem(BaseModel):
    """WebSocket内容项"""
    type: str  # text/file/audio
    content: str
    files: Optional[List[dict]] = None


class WebSocketMessageData(BaseModel):
    """WebSocket消息数据"""
    role: str  # user/assistant
    aiModel: str = ""
    content: List[WebSocketContentItem]


class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    type: str  # heartbeat/aiMessage/tts
    data: WebSocketMessageData
    done: bool = False
