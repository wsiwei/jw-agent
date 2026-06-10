import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional

from app.config import config
from langgraph.errors import GraphInterrupt
from app.models import (
    AgentGetRecordRequest, AgentGetRecordResponse,
    SpeechTTSRequest, SpeechTTSResponse,
    SpeechASRRequest, SpeechASRResponse,
    WebSocketMessage, WebSocketMessageData, WebSocketContentItem, AiChatRecord,
    KnowledgeCategoryRequest, KnowledgeDocumentRequest,
    KnowledgeWriteRequest, KnowledgeDeleteRequest,
)
from app.agent_service import AgentService
from app.speech_service import SpeechService
from app.knowledge_service import get_knowledge_service
from app.utils import success_response, error_response, generate_chat_id, get_current_timestamp, generate_uuid
from app.database import init_database

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建应用
app = FastAPI(
    title="Agent Service",
    description="汕尾市公权力大数据监督平台Agent服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
upload_path = Path(config.upload_path)
upload_path.mkdir(exist_ok=True)
app.mount(f"/{config.upload_path}", StaticFiles(directory=config.upload_path), name="upload")

# 初始化服务
agent_service = AgentService()
speech_service = SpeechService()

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"客户端已连接: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"客户端已断开: {client_id}")

manager = ConnectionManager()


# ==================== Agent 接口 ====================

@app.post(f"{config.base_path}/agent/record/get")
async def get_agent_record(request: AgentGetRecordRequest):
    """获取会话记录"""
    try:
        records = await agent_service.get_chat_records(
            request.session_id,
            request.cursor,
            request.limit
        )
        return success_response({"list": [r.dict() for r in records]})
    except Exception as e:
        logger.error(f"获取会话记录失败: {e}")
        return JSONResponse(
            status_code=500,
            content=error_response(500, "查询失败")
        )


@app.websocket(f"{config.base_path}/agent/chat")
async def agent_chat(websocket: WebSocket, user_id: str = "", session_id: str = ""):
    """WebSocket对话接口"""
    import asyncio
    from app.utils import generate_uuid
    client_id = generate_uuid()

    await manager.connect(websocket, client_id)

    # 当前正在运行的 stream_chat task，用于支持中止
    current_task: asyncio.Task = None

    try:
        while True:
            data = await websocket.receive_json()
            message = WebSocketMessage(**data)

            logger.info(f"收到消息类型: {message.type}, SessionID: {session_id}")

            if message.type == "heartbeat":
                continue

            elif message.type == "aiMessage":
                if len(message.data.content) == 0:
                    continue

                # abort 控制消息：取消当前正在运行的 task
                if message.data.content[0].type == "abort":
                    logger.info(f"[WebSocket] 收到 abort 消息，取消当前任务")
                    if current_task and not current_task.done():
                        current_task.cancel()
                        agent_service.cleanup_session(session_id)
                    continue

                user_content = message.data.content[0].content
                domain = (message.data.content[0].extend or {}).get("domain", "")
                ai_model = message.data.aiModel or "glm-4"
                record_id = message.data.content[0].record_id or ""
                user_roles = message.data.userRole or []
                logger.info(f"[请求] SessionID={session_id} UserID={user_id} Roles={user_roles} Domain={domain!r} Content={user_content[:50]!r}")

                # 保存用户消息
                user_record = AiChatRecord(
                    chat_id=generate_chat_id(session_id),
                    user_id=user_id,
                    session_id=session_id,
                    ai_model=ai_model,
                    role="user",
                    content=user_content,
                    create_time=get_current_timestamp()
                )
                await agent_service.save_chat_record(user_record)

                # 以 Task 方式运行，支持中止
                async def run_chat():
                    try:
                        await agent_service.stream_chat(websocket, user_content, ai_model, session_id, domain=domain, record_id=record_id, user_roles=user_roles)
                    except GraphInterrupt:
                        logger.info("[AgentService] 图执行已挂起（interrupt），等待用户回复")
                    except asyncio.CancelledError:
                        logger.info("[AgentService] 任务已被中止")
                        agent_service.cleanup_session(session_id)
                    except Exception as e:
                        logger.error(f"处理消息失败: {e}", exc_info=True)
                        try:
                            err_msg = WebSocketMessage(
                                type="aiMessage",
                                data=WebSocketMessageData(
                                    role="assistant",
                                    aiModel=ai_model,
                                    content=[WebSocketContentItem(type="text", content=f"处理失败：{e}")]
                                ),
                                done=True
                            )
                            await websocket.send_json(err_msg.dict())
                        except Exception:
                            pass

                current_task = asyncio.create_task(run_chat())

            elif message.type == "tts":
                if len(message.data.content) == 0:
                    continue
                text = message.data.content[0].content
                await agent_service.stream_tts(websocket, text)

    except WebSocketDisconnect:
        if current_task and not current_task.done():
            current_task.cancel()
        manager.disconnect(client_id)

    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(client_id)



# ==================== Speech 接口 ====================

@app.post(f"{config.base_path}/speech/tts")
async def speech_tts(request: SpeechTTSRequest):
    """语音合成（TTS）"""
    try:
        audio_url = await speech_service.text_to_speech(
            request.text,
            request.voice,
            request.speed
        )
        return success_response({"audio_url": audio_url})
    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        return JSONResponse(
            status_code=500,
            content=error_response(1007, "语音合成失败")
        )


@app.post(f"{config.base_path}/speech/asr")
async def speech_asr(request: SpeechASRRequest):
    """语音识别（ASR）"""
    try:
        text = await speech_service.speech_to_text(request.encode)
        return success_response({"text": text})
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        return JSONResponse(
            status_code=500,
            content=error_response(1006, "语音识别失败")
        )


# ==================== Knowledge 接口 ====================

@app.post(f"{config.base_path}/knowledge/category")
async def get_knowledge_category(request: KnowledgeCategoryRequest):
    """获取知识库分类列表（当前用 Qdrant collection 内的 category_id 聚合）"""
    try:
        ks = get_knowledge_service()
        # 滚动查询所有 payload，聚合出不重复的 category_id
        categories = set()
        offset = None
        while True:
            result, offset = ks._client.scroll(
                collection_name=ks._collection,
                limit=1000,
                offset=offset,
                with_payload=["category_id", "source"],
            )
            for point in result:
                cid = point.payload.get("category_id", "")
                if cid:
                    categories.add(cid)
            if offset is None:
                break
        return success_response([{"category_id": c} for c in sorted(categories)])
    except Exception as e:
        logger.error(f"获取知识库分类失败: {e}")
        return success_response([])


@app.post(f"{config.base_path}/knowledge/document")
async def get_knowledge_document(request: KnowledgeDocumentRequest):
    """获取某分类下的文档列表"""
    try:
        ks = get_knowledge_service()
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        docs = {}
        offset = None
        while True:
            result, offset = ks._client.scroll(
                collection_name=ks._collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="category_id",
                                   match=MatchValue(value=request.category_id))
                ]) if request.category_id else None,
                limit=1000,
                offset=offset,
                with_payload=["doc_id", "source", "create_time"],
            )
            for point in result:
                doc_id = point.payload.get("doc_id", "")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "source": point.payload.get("source", ""),
                        "create_time": point.payload.get("create_time", 0),
                    }
            if offset is None:
                break
        return success_response({"documents": list(docs.values())})
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return success_response({"documents": []})


@app.post(f"{config.base_path}/knowledge/document/write")
async def write_knowledge_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    category_id: str = Form(...),
    chunk_size: int = Form(500),
    overlap: int = Form(50),
):
    """上传文件并切分写入 Qdrant"""
    try:
        upload_dir = Path(config.upload_path) / "knowledge"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename
        file_path.write_bytes(await file.read())

        ks = get_knowledge_service()
        count = ks.ingest_file(str(file_path), doc_id, category_id, chunk_size, overlap)
        return success_response({"doc_id": doc_id, "chunks": count})
    except Exception as e:
        logger.error(f"写入知识库失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content=error_response(500, str(e)))


@app.post(f"{config.base_path}/knowledge/document/delete")
async def delete_knowledge_document(request: KnowledgeDeleteRequest):
    """删除文档的所有 chunk"""
    try:
        ks = get_knowledge_service()
        ks.delete_document(request.doc_id)
        return success_response({"doc_id": request.doc_id})
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        return JSONResponse(status_code=500, content=error_response(500, str(e)))


# ==================== 健康检查 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "agent-sw-python"
    }


# ==================== 启动信息 ====================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Agent 服务启动中...")
    logger.info("=" * 60)

    # 初始化数据库
    init_database()

    logger.info("=" * 60)
    logger.info("Agent 服务启动成功")
    logger.info(f"服务地址: http://{config.host}:{config.port}")
    logger.info(f"健康检查: http://{config.host}:{config.port}/health")
    logger.info(f"API文档: http://{config.host}:{config.port}/docs")
    logger.info(f"WebSocket: ws://{config.host}:{config.port}{config.base_path}/agent/chat")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=True
    )
