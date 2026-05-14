import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.config import config
from app.database import db
from app.models import AiChatRecord, WebSocketContentItem, WebSocketMessage, WebSocketMessageData
from app.utils import generate_chat_id, get_current_timestamp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 启动时加载静态资源
# ─────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent
_TABLE_LIST = (_BASE / "db_table_list.txt").read_text(encoding='utf-8') \
    if (_BASE / "db_table_list.txt").exists() else ""
_COL_MAP: dict = json.loads((_BASE / "db_columns.json").read_text(encoding='utf-8')) \
    if (_BASE / "db_columns.json").exists() else {}

_PERSON_REPORT_PROMPT = (_BASE / "prompt" / "person_report.txt").read_text(encoding='utf-8') \
    if (_BASE / "prompt" / "person_report.txt").exists() else ""

_PERSON_REPORT_TABLES = [
    (item["table"], item["label"], item["id_col"])
    for item in json.loads(
        (_BASE / "config" / "person_report.json").read_text(encoding='utf-8')
    )["tables"]
]

_BASE_PERSON_TABLES = [
    ("DIMENSYSTEM.PERSONNEL_PROFILE",  "全市公职人员数据",      "IDCARD",   "NAME"),
    ("DIMENSYSTEM.PUBLIC_INSTITUTION", "全市事业干部人员数据",   "ID_CARD",  "NAME"),
]
_BASE_TABLE_NAMES = {t[0].lower() for t in _BASE_PERSON_TABLES}

# ─────────────────────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────────────────────
_SELECT_TABLE_PROMPT = f"""你是一个数据库专家。根据用户的问题，从以下表列表中选出最相关的表，返回表的完整名称（schema.表名），每行一个，不要任何解释。

{_TABLE_LIST}

选表规则：
1. 优先选能单表回答问题的表，不要为了"更完整"而多选
2. 只有当问题明确需要多表数据时才选多张表（最多3张）
3. 不要选注释为空或与问题无关的表
4. Schema优先级：DIMENSYSTEM > SWYD > DIMENMODEL > SYSDBA
5. 如果完全找不到相关表，只返回：UNKNOWN
"""

_GEN_SQL_PROMPT_TPL = """你是一个达梦数据库SQL专家。根据用户问题和以下表结构，生成对应的SQL查询语句。

{schema_info}

要求：
1. 只返回SQL语句，不要任何解释文字
2. SQL语句以分号结尾
3. 表名必须带schema前缀（如 DIMENSYSTEM.PERSONNEL_PROFILE）
4. 默认加 LIMIT 100
5. 优先用单表查询，只有字段确实分布在多张表时才 JOIN
6. JOIN 前确认关联字段在两张表中都存在且有实际数据
"""


# ─────────────────────────────────────────────────────────────
# 人员报告字段描述（启动时构建）
# ─────────────────────────────────────────────────────────────
def _build_person_report_schema() -> Dict[str, str]:
    """为 person_report.json 中每张表构建字段描述，供 AI 参考。"""
    schema_map: Dict[str, str] = {}
    target_tables = {item["table"].upper() for item in
                     json.loads((_BASE / "config" / "person_report.json").read_text(encoding='utf-8'))["tables"]}

    compact_path = _BASE / "db_schema_compact.txt"
    if compact_path.exists():
        text = compact_path.read_text(encoding='utf-8')
        blocks = re.split(r'(?=^表: )', text, flags=re.MULTILINE)
        for block in blocks:
            m = re.match(r'^表: (\S+)', block)
            if not m:
                continue
            tbl = m.group(1).upper()
            if tbl in target_tables:
                schema_map[tbl.lower()] = block.strip()

    missing = [t for t in target_tables if t.lower() not in schema_map]
    if missing:
        try:
            for full_name in missing:
                parts = full_name.split(".")
                if len(parts) != 2:
                    continue
                owner, tbl_name = parts[0], parts[1]
                rows = db.execute_query(
                    "SELECT COLUMN_NAME, DATA_TYPE FROM ALL_TAB_COLUMNS "
                    "WHERE OWNER = ? AND TABLE_NAME = ? ORDER BY COLUMN_ID",
                    (owner, tbl_name)
                )
                if rows:
                    lines = [f"表: {full_name}"]
                    lines += [f"  {r['COLUMN_NAME']} {r['DATA_TYPE']}" for r in rows]
                    schema_map[full_name.lower()] = "\n".join(lines)
        except Exception as e:
            logger.warning(f"[人员报告] 查询 ALL_TAB_COLUMNS 失败: {e}")

    logger.info(f"[人员报告] 字段描述加载完成，共 {len(schema_map)} 张表")
    return schema_map


_PERSON_REPORT_SCHEMA: Dict[str, str] = _build_person_report_schema()


# ─────────────────────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    session_id: str
    ai_model: str
    content: str        # 当前用户输入
    intent: str         # 意图分类结果
    name: str           # 提取的人名
    id_card: str        # 确认后的身份证号
    candidates: list    # 重名候选列表
    base_data: dict     # 基础表查询结果
    all_data: dict      # 全部表查询结果
    sql: str            # 生成的 SQL
    query_results: list # SQL 执行结果
    report: str         # 最终报告文本


# ─────────────────────────────────────────────────────────────
# AgentService
# ─────────────────────────────────────────────────────────────
class AgentService:

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=config.ai.get('api_key'),
            base_url=config.ai.get('base_url'),
            model=config.ai.get('model'),
            temperature=0,
            timeout=60,
            max_retries=1,
        )
        fast_model = config.ai.get('fast_model', config.ai.get('model'))
        self.fast_llm = ChatOpenAI(
            api_key=config.ai.get('api_key'),
            base_url=config.ai.get('base_url'),
            model=fast_model,
            temperature=0,
            timeout=60,
            max_retries=1,
        )
        # WebSocket 注册表（不进 graph state，避免序列化问题）
        self._ws_registry: Dict[str, Any] = {}
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    # ── 聊天记录 ──────────────────────────────────────────────

    async def get_chat_records(self, session_id: str, cursor: int, limit: int) -> List[AiChatRecord]:
        try:
            records_data = db.get_chat_records(session_id, limit, cursor)
            records = []
            for data in records_data:
                record = AiChatRecord(
                    chat_id=data.get('CHAT_ID'),
                    user_id=data.get('USER_ID', ''),
                    session_id=data.get('SESSION_ID'),
                    ai_model=data.get('AI_MODEL'),
                    role=data.get('ROLE'),
                    content=data.get('CONTENT'),
                    user_prompt=data.get('USER_PROMPT'),
                    system_prompt=data.get('SYSTEM_PROMPT'),
                    create_time=data.get('CREATE_TIME'),
                    prompt_tokens=data.get('PROMPT_TOKENS', 0),
                    completion_tokens=data.get('COMPLETION_TOKENS', 0),
                    total_tokens=data.get('TOTAL_TOKENS', 0)
                )
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"获取聊天记录失败: {e}")
            return []

    async def save_chat_record(self, record: AiChatRecord):
        logger.info(f"保存聊天记录: SessionID={record.session_id}, Role={record.role}")
        try:
            db.insert_chat_record({
                'chat_id': record.chat_id,
                'user_id': record.user_id,
                'session_id': record.session_id,
                'ai_model': record.ai_model,
                'role': record.role,
                'content': record.content,
                'user_prompt': record.user_prompt,
                'system_prompt': record.system_prompt,
                'create_time': record.create_time,
                'prompt_tokens': record.prompt_tokens,
                'completion_tokens': record.completion_tokens,
                'total_tokens': record.total_tokens,
            })
        except Exception as e:
            logger.error(f"保存聊天记录失败: {e}")

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _extract_sql(text: str) -> str:
        match = re.search(r'```(?:sql)?\s*([\s\S]+?)```', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _mask_idcard(idcard: str) -> str:
        if len(idcard) >= 10:
            return idcard[:6] + "****" + idcard[-4:]
        return idcard[:2] + "****" if len(idcard) > 2 else "****"

    @staticmethod
    def _build_candidate_message(candidates: list) -> str:
        lines = [f"找到 {len(candidates)} 位同名人员，请回复序号选择："]
        for i, c in enumerate(candidates, 1):
            unit = c.get("unit", "")
            pos = c.get("position", "")
            masked = AgentService._mask_idcard(c["idcard"])
            lines.append(f"{i}. {c['name']} — {unit}{'—' + pos if pos else ''}（身份证：{masked}）")
        return "\n".join(lines)

    def _query_person_data(self, id_card: str, skip_tables: Optional[set] = None) -> dict:
        """按身份证号查询 person_report.json 中的表，skip_tables 中的表跳过（小写表名）"""
        skip_tables = skip_tables or set()
        results = {}
        for table, label, id_col in _PERSON_REPORT_TABLES:
            if table.lower() in skip_tables:
                continue
            try:
                rows = db.execute_query(
                    f"SELECT * FROM {table} WHERE {id_col} = ? LIMIT 50",
                    (id_card,)
                )
                if rows:
                    results[label] = rows
                    logger.info(f"  {label}: {len(rows)} 条")
                else:
                    logger.info(f"  {label}: 无数据")
            except Exception as e:
                logger.warning(f"  {label} 查询失败: {e}")
        return results

    def _format_results(self, results: list) -> str:
        if not results:
            return "无数据"
        columns = list(results[0].keys())
        lines = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---" for _ in columns]) + "|",
        ]
        for row in results:
            values = [str(row.get(col, '')) for col in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    # ── WebSocket 推送 ────────────────────────────────────────

    async def _send_progress(self, websocket, ai_model: str, text: str):
        if not websocket:
            return
        msg = WebSocketMessage(
            type="progress",
            data=WebSocketMessageData(
                role="assistant",
                aiModel=ai_model,
                content=[WebSocketContentItem(type="text", content=text)]
            ),
            done=False
        )
        await websocket.send_json(msg.dict())

    async def _stream_response(self, websocket, response: str, ai_model: str):
        chunk_size = 20
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            done = (i + chunk_size) >= len(response)
            message = WebSocketMessage(
                type="aiMessage",
                data=WebSocketMessageData(
                    role="assistant",
                    aiModel=ai_model,
                    content=[WebSocketContentItem(type="text", content=chunk)]
                ),
                done=done
            )
            await websocket.send_json(message.dict())
            await asyncio.sleep(0.02)

    async def _stream_ai_to_ws(self, prompt: str, websocket, ai_model: str,
                               temperature: float = 0.3) -> str:
        """流式调用 AI，边生成边推送到 WebSocket，返回完整文本（用于存库）。"""
        llm = self.llm.bind(temperature=temperature)
        full_text = ""
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            delta = chunk.content or ""
            if delta:
                full_text += delta
                if websocket:
                    await websocket.send_json(WebSocketMessage(
                        type="aiMessage",
                        data=WebSocketMessageData(
                            role="assistant",
                            aiModel=ai_model,
                            content=[WebSocketContentItem(type="text", content=delta)]
                        ),
                        done=False
                    ).dict())
        if websocket:
            await websocket.send_json(WebSocketMessage(
                type="aiMessage",
                data=WebSocketMessageData(
                    role="assistant",
                    aiModel=ai_model,
                    content=[WebSocketContentItem(type="text", content="")]
                ),
                done=True
            ).dict())
        logger.info(f"[流式输出] 完成，共 {len(full_text)} 字符")
        return full_text

    # ── LangGraph 节点 ────────────────────────────────────────

    async def _node_classify_intent(self, state: AgentState) -> dict:
        prompt = (
            "你是一个意图分类器。根据用户输入，判断属于以下哪种意图，"
            "只返回对应的英文标识，不要任何解释：\n\n"
            "- person_report：查询某个具体人员的综合信息、档案、报告\n"
            "  （如：张三的人员信息报告、查一下李四的档案、王五的个人情况）\n"
            "- nl2sql：查询数据库中的结构化数据，如统计数量、列表、排名、金额等\n"
            "  （如：财政局有多少科室、查询所有局长、汕尾市公职人员总数）\n"
            "- knowledge：询问概念定义、政策法规、操作方法等知识性问题\n"
            "  （如：什么是三公经费、如何申请行政许可）\n"
            "- normal：其他普通对话\n\n"
            f"用户输入：{state['content']}"
        )
        valid_intents = {"person_report", "nl2sql", "knowledge", "normal"}
        try:
            resp = await asyncio.to_thread(
                self.fast_llm.invoke,
                [HumanMessage(content=prompt)]
            )
            intent = resp.content.strip().lower()
            if intent not in valid_intents:
                logger.warning(f"[意图识别] 模型返回未知意图 '{intent}'，降级为 nl2sql")
                intent = "nl2sql"
        except Exception as e:
            logger.error(f"[意图识别] AI 调用失败: {e}，降级为 nl2sql")
            intent = "nl2sql"
        logger.info(f"[意图识别] {intent}")
        return {"intent": intent}

    async def _node_extract_name(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        if ws:
            await self._send_progress(ws, state["ai_model"], "🔍 正在识别人员信息...")
        try:
            resp = await asyncio.to_thread(
                self.fast_llm.invoke,
                [HumanMessage(content=(
                    "从以下文本中提取被查询人的姓名。"
                    "注意：'人员信息报告'、'人员报告'、'信息报告'、'综合报告'等是固定词组，不是人名的一部分。"
                    "只返回人名，没有则返回空字符串。\n"
                    f"文本：{state['content']}"
                ))]
            )
            name = resp.content.strip()
        except Exception as e:
            logger.error(f"[人员报告] 提取人名失败: {e}")
            name = ""
        logger.info(f"[人员报告] 目标人员: {name}")
        return {"name": name}

    async def _node_lookup_person(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        name = state["name"]
        if not name:
            if ws:
                await self._stream_response(ws, "请告诉我要查询哪位人员的信息报告，例如：生成张三的人员信息报告", state["ai_model"])
            return {"candidates": []}
        if ws:
            await self._send_progress(ws, state["ai_model"], f"📋 正在查询 {name} 的基本信息...")
        try:
            rows_profile = await asyncio.to_thread(
                db.execute_query,
                "SELECT * FROM DIMENSYSTEM.PERSONNEL_PROFILE WHERE NAME = ? LIMIT 10",
                (name,)
            )
            rows_institution = await asyncio.to_thread(
                db.execute_query,
                "SELECT * FROM DIMENSYSTEM.PUBLIC_INSTITUTION WHERE NAME = ? LIMIT 10",
                (name,)
            )
        except Exception as e:
            if ws:
                await self._stream_response(ws, f"查询失败：{e}", state["ai_model"])
            return {"candidates": []}

        candidates = []
        seen_idcards: set = set()
        for row in (rows_profile or []):
            ic = row.get("IDCARD", "")
            if ic and ic not in seen_idcards:
                seen_idcards.add(ic)
                candidates.append({
                    "name": row.get("NAME", name), "idcard": ic,
                    "unit": row.get("UNIT", ""), "position": row.get("CURRENTPOSITION", ""),
                    "source": "DIMENSYSTEM.PERSONNEL_PROFILE",
                    "label": "全市公职人员数据", "row": row,
                })
        for row in (rows_institution or []):
            ic = row.get("ID_CARD", "")
            if ic and ic not in seen_idcards:
                seen_idcards.add(ic)
                candidates.append({
                    "name": row.get("NAME", name), "idcard": ic,
                    "unit": row.get("UNIT", ""), "position": row.get("POSITION", ""),
                    "source": "DIMENSYSTEM.PUBLIC_INSTITUTION",
                    "label": "全市事业干部人员数据", "row": row,
                })

        if not candidates and ws:
            await self._stream_response(ws, f"未找到 {name} 的人员信息，请确认姓名是否正确。", state["ai_model"])
            return {"candidates": []}

        # 单结果时直接填充 id_card 和 base_data，省去 disambiguate 节点
        if len(candidates) == 1:
            chosen = candidates[0]
            return {
                "candidates": candidates,
                "id_card": chosen["idcard"],
                "base_data": {chosen["label"]: [chosen["row"]]},
            }
        return {"candidates": candidates}

    async def _node_disambiguate(self, state: AgentState) -> dict:
        candidates = state["candidates"]
        ws = self._ws_registry.get(state["session_id"])
        msg = self._build_candidate_message(candidates)
        logger.info(f"[人员报告] 发现重名 {len(candidates)} 人，等待用户确认")
        if ws:
            await self._stream_response(ws, msg, state["ai_model"])

        # 挂起，等待用户回复
        user_choice = interrupt({"candidates": candidates})

        # 恢复后解析序号
        m = re.search(r'\d+', str(user_choice).strip())
        if not m:
            if ws:
                await self._stream_response(ws, "请回复序号（如：1）\n\n" + msg, state["ai_model"])
            user_choice = interrupt({"candidates": candidates})
            m = re.search(r'\d+', str(user_choice).strip())

        idx = (int(m.group()) - 1) if m else 0
        idx = max(0, min(idx, len(candidates) - 1))
        chosen = candidates[idx]
        logger.info(f"[人员报告] 用户选择第 {idx+1} 位：{chosen['name']} ({chosen['idcard']})")
        return {
            "id_card": chosen["idcard"],
            "base_data": {chosen["label"]: [chosen["row"]]},
        }

    async def _node_generate_report(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        ai_model = state["ai_model"]
        name = state["name"]
        id_card = state["id_card"]
        base_data = state.get("base_data") or {}

        async def progress(text):
            if ws:
                await self._send_progress(ws, ai_model, text)

        await progress(f"🗄️ 正在从多个数据源查询 {name} 的信息...")
        other_data = await asyncio.to_thread(
            self._query_person_data, id_card, _BASE_TABLE_NAMES
        )
        all_data = {**base_data, **other_data}

        data_sections = []
        for label, rows in all_data.items():
            cols = list(rows[0].keys())
            lines = ["| " + " | ".join(cols) + " |",
                     "|" + "|".join(["---"] * len(cols)) + "|"]
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(c, '')) for c in cols) + " |")
            data_sections.append(f"### {label}\n" + "\n".join(lines))
        data_text = "\n\n".join(data_sections) if data_sections else "未查询到任何数据"

        schema_parts = []
        for table, label, _ in _PERSON_REPORT_TABLES:
            schema_txt = _PERSON_REPORT_SCHEMA.get(table.lower())
            if schema_txt:
                schema_parts.append(f"[{label}]\n{schema_txt}")
        schema_context = "\n\n".join(schema_parts) if schema_parts else ""

        await progress("✍️ 正在生成人员信息报告...")
        prompt = _PERSON_REPORT_PROMPT.replace("{question}", name).replace("{answer}", data_text)
        if schema_context:
            prompt = f"以下是各数据表的字段说明，供参考：\n\n{schema_context}\n\n---\n\n{prompt}"

        logger.info("=" * 60)
        logger.info(f"[人员报告] 开始流式生成，数据章节数: {len(all_data)}，prompt 长度: {len(prompt)} 字符")

        try:
            report = await self._stream_ai_to_ws(prompt, ws, ai_model)
        except Exception as e:
            logger.error(f"[人员报告] 流式生成失败: {e}", exc_info=True)
            report = f"## {name} 人员信息（原始数据）\n\nAI生成报告失败（{type(e).__name__}），以下为数据库查询结果：\n\n{data_text}"
            if ws:
                await self._stream_response(ws, report, ai_model)

        logger.info("=" * 60)
        return {"report": report, "all_data": all_data}

    async def _node_select_tables(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        if ws:
            await self._send_progress(ws, state["ai_model"], "🔍 正在分析问题，生成查询SQL...")
        logger.info("=" * 60)
        logger.info("[节点1a: 选表] 输入 >>>")
        logger.info(f"  user: {state['content']}")
        try:
            resp = await asyncio.to_thread(
                self.llm.invoke,
                [SystemMessage(content=_SELECT_TABLE_PROMPT),
                 HumanMessage(content=state["content"])]
            )
            raw_tables = resp.content.strip()
        except Exception as e:
            logger.error(f"[选表] 失败: {e}")
            raw_tables = "UNKNOWN"
        logger.info(f"[节点1a: 选表] 输出 <<< 选中表: {raw_tables}")
        return {"sql": raw_tables}   # 暂存选表结果，generate_sql 节点读取

    async def _node_generate_sql(self, state: AgentState) -> dict:
        raw_tables = state.get("sql", "")
        if not raw_tables or raw_tables.upper() == "UNKNOWN":
            return {"sql": "UNKNOWN"}
        selected = [t.strip() for t in raw_tables.splitlines() if t.strip()]
        schema_parts = []
        for tbl in selected:
            cols = _COL_MAP.get(tbl, [])
            if cols:
                schema_parts.append(f"表: {tbl}\n" + "\n".join(cols))
            else:
                schema_parts.append(f"表: {tbl}  (无字段信息)")
        schema_info = "\n\n".join(schema_parts)
        system2 = _GEN_SQL_PROMPT_TPL.format(schema_info=schema_info)
        logger.info(f"[节点1b: 生成SQL] 输入 >>> 选中表: {selected}")
        try:
            resp = await asyncio.to_thread(
                self.llm.invoke,
                [SystemMessage(content=system2),
                 HumanMessage(content=state["content"])]
            )
            raw_sql = resp.content.strip()
        except Exception as e:
            logger.error(f"[生成SQL] 失败: {e}")
            raw_sql = "UNKNOWN"
        sql = self._extract_sql(raw_sql)
        logger.info(f"[节点1b: 生成SQL] 输出 <<< {sql}")
        logger.info("=" * 60)
        return {"sql": sql}

    async def _node_execute_sql(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        sql = state.get("sql", "")
        ai_model = state["ai_model"]
        if not sql or sql.upper() == "UNKNOWN":
            if ws:
                await self._stream_response(ws, "抱歉，无法识别您的查询意图，请描述得更具体一些。", ai_model)
            return {"query_results": []}
        if ws:
            await self._send_progress(ws, ai_model, "📋 已生成SQL，正在查询数据库...")

        results = []
        sql_error = None
        for attempt in range(2):
            logger.info("=" * 60)
            logger.info(f"[节点2: 数据库查询] 执行{'（重试）' if attempt > 0 else ''} >>> SQL: {sql}")
            try:
                results = await asyncio.to_thread(db.execute_query, sql)
                logger.info(f"[节点2: 数据库查询] 结果 <<< 返回 {len(results)} 条记录")
                logger.info("=" * 60)
                sql_error = None
                if results:
                    break
                if attempt == 0:
                    if ws:
                        await self._send_progress(ws, ai_model, "⚠️ 查询无结果，正在优化查询条件...")
                    sql = await asyncio.to_thread(
                        self._fix_sql, state["content"], sql, "查询返回0条记录，可能是JOIN条件不匹配，请简化为单表查询"
                    )
            except Exception as e:
                sql_error = str(e)
                logger.error(f"[节点2: 数据库查询] 失败: {e}")
                logger.info("=" * 60)
                if attempt == 0:
                    if ws:
                        await self._send_progress(ws, ai_model, "⚠️ SQL执行出错，正在修正...")
                    sql = await asyncio.to_thread(self._fix_sql, state["content"], sql, f"执行报错：{e}")

        if (sql_error and not results) or not results:
            msg = "查询失败，请换个方式描述问题。" if sql_error else "查询无结果，数据库中暂无符合条件的数据。"
            if ws:
                await self._stream_response(ws, msg, ai_model)
        return {"query_results": results, "sql": sql}

    async def _node_answer_with_data(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        ai_model = state["ai_model"]
        results = state.get("query_results") or []
        if not results:
            return {"report": ""}
        if ws:
            await self._send_progress(ws, ai_model, f"✅ 查询到 {len(results)} 条数据，正在分析结果...")
        sample = results[:50]
        data_text = self._format_results(sample)
        total = len(results)
        truncated_note = f"（数据共 {total} 条，以下展示前 {len(sample)} 条）" if total > 50 else f"（共 {total} 条）"
        prompt = f"""用户问题：{state['content']}

已从数据库查询到以下数据{truncated_note}：

{data_text}

请根据以上数据，用简洁清晰的中文回答用户的问题。如果数据中有明确的列表信息，可以用表格或列表展示关键字段。"""
        logger.info("=" * 60)
        logger.info(f"[节点3: 数据分析] 输入 >>> 数据行数: {total}，传给AI: {len(sample)} 条")
        answer = await self._stream_ai_to_ws(prompt, ws, ai_model, temperature=0.3)
        logger.info(f"[节点3: 数据分析] 输出 <<< 长度: {len(answer)} 字符")
        logger.info("=" * 60)
        return {"report": answer}

    async def _node_normal_chat(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        logger.info("=" * 60)
        logger.info(f"[节点1: 普通对话] 输入 >>> {state['content']}")
        answer = await self._stream_ai_to_ws(state["content"], ws, state["ai_model"], temperature=0.7)
        logger.info(f"[节点1: 普通对话] 输出 <<< 长度: {len(answer)} 字符")
        logger.info("=" * 60)
        return {"report": answer}

    async def _node_knowledge(self, state: AgentState) -> dict:
        ws = self._ws_registry.get(state["session_id"])
        response = "这是知识库查询的回复内容。"
        if ws:
            await self._stream_response(ws, response, state["ai_model"])
        return {"report": response}

    def _fix_sql(self, question: str, bad_sql: str, error_hint: str) -> str:
        prompt = f"""以下SQL执行出现问题，请修正后返回正确的SQL（只返回SQL，不要解释）。

用户问题：{question}

原SQL：
{bad_sql}

问题：{error_hint}

修正要求：
- 优先改为单表查询，去掉不必要的 JOIN
- 保持 LIMIT 100
- 表名带 schema 前缀
"""
        logger.info(f"[SQL修正] 输入: {error_hint}")
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        fixed = self._extract_sql(resp.content.strip())
        logger.info(f"[SQL修正] 输出: {fixed}")
        return fixed

    # ── Graph 构建 ────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("classify_intent",  self._node_classify_intent)
        g.add_node("extract_name",     self._node_extract_name)
        g.add_node("lookup_person",    self._node_lookup_person)
        g.add_node("disambiguate",     self._node_disambiguate)
        g.add_node("generate_report",  self._node_generate_report)
        g.add_node("select_tables",    self._node_select_tables)
        g.add_node("generate_sql",     self._node_generate_sql)
        g.add_node("execute_sql",      self._node_execute_sql)
        g.add_node("answer_with_data", self._node_answer_with_data)
        g.add_node("normal_chat",      self._node_normal_chat)
        g.add_node("knowledge",        self._node_knowledge)

        g.set_entry_point("classify_intent")

        def route_intent(state: AgentState) -> str:
            return state.get("intent", "normal")

        g.add_conditional_edges("classify_intent", route_intent, {
            "nl2sql":        "select_tables",
            "person_report": "extract_name",
            "knowledge":     "knowledge",
            "normal":        "normal_chat",
        })

        g.add_edge("extract_name", "lookup_person")

        def route_person_lookup(state: AgentState) -> str:
            candidates = state.get("candidates") or []
            if not candidates:
                return "not_found"
            if len(candidates) == 1:
                return "single"
            return "multiple"

        g.add_conditional_edges("lookup_person", route_person_lookup, {
            "single":    "generate_report",
            "multiple":  "disambiguate",
            "not_found": END,
        })

        # 单结果路径：lookup_person 直接进 generate_report 前需设置 id_card 和 base_data
        # 通过 _node_lookup_person 在 single 分支时返回这两个字段
        g.add_edge("disambiguate",     "generate_report")
        g.add_edge("generate_report",  END)
        g.add_edge("select_tables",    "generate_sql")
        g.add_edge("generate_sql",     "execute_sql")
        g.add_edge("execute_sql",      "answer_with_data")
        g.add_edge("answer_with_data", END)
        g.add_edge("normal_chat",      END)
        g.add_edge("knowledge",        END)

        compiled = g.compile(checkpointer=self.checkpointer)
        logger.info("[AgentService] StateGraph compiled successfully")
        return compiled

    # ── 主入口 ────────────────────────────────────────────────

    async def stream_chat(self, websocket, content: str, ai_model: str, session_id: str):
        logger.info("=" * 60)
        logger.info(f"[用户输入] {content}")

        self._ws_registry[session_id] = websocket
        graph_config = {"configurable": {"thread_id": session_id}}

        try:
            state_snapshot = self.graph.get_state(graph_config)
            if state_snapshot.next:
                logger.info("[意图识别] 恢复 interrupt（重名确认）")
                result = await self.graph.ainvoke(Command(resume=content), graph_config)
            else:
                result = await self.graph.ainvoke(
                    {
                        "session_id": session_id,
                        "ai_model": ai_model,
                        "content": content,
                        "intent": "", "name": "", "id_card": "",
                        "candidates": [], "base_data": {}, "all_data": {},
                        "sql": "", "query_results": [], "report": "",
                    },
                    graph_config
                )
        finally:
            self._ws_registry.pop(session_id, None)

        response = (result or {}).get("report") or ""
        assistant_record = AiChatRecord(
            chat_id=generate_chat_id(session_id),
            session_id=session_id,
            ai_model=ai_model,
            role="assistant",
            content=response,
            create_time=get_current_timestamp(),
        )
        await self.save_chat_record(assistant_record)
