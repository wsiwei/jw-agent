"""
达梦数据库连接模块
"""

import dmPython
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from app.config import config

logger = logging.getLogger(__name__)


class DMDatabase:
    """达梦数据库管理类"""

    def __init__(self):
        self.host = config.database.get('host')
        self.port = config.database.get('port')
        self.database = config.database.get('database')
        self.user = config.database.get('user')
        self.password = config.database.get('password')
        self._connection = None

    def get_connection_string(self) -> str:
        """获取连接字符串"""
        return f"{self.user}/{self.password}@{self.host}:{self.port}"

    def connect(self):
        """建立数据库连接"""
        try:
            logger.info(f"正在连接达梦数据库: {self.host}:{self.port}")
            self._connection = dmPython.connect(
                user=self.user,
                password=self.password,
                server=self.host,
                port=self.port
            )
            logger.info("✅ 达梦数据库连接成功")
            return self._connection
        except Exception as e:
            logger.error(f"❌ 达梦数据库连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            logger.info("数据库连接已关闭")

    @contextmanager
    def get_cursor(self):
        """获取游标（上下文管理器）"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def execute_query(self, sql: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询并返回结果"""
        with self.get_cursor() as cursor:
            logger.info(f"执行SQL: {sql}")
            if params:
                logger.info(f"参数: {params}")
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            # 获取列名
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # 获取结果
            rows = cursor.fetchall()

            # 转换为字典列表
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))

            logger.info(f"查询返回 {len(results)} 条记录")
            if results:
                preview = results[:3]
                for i, row in enumerate(preview):
                    logger.info(f"  [{i+1}] {row}")
                if len(results) > 3:
                    logger.info(f"  ... 共 {len(results)} 条，仅展示前3条")
            return results

    def execute_update(self, sql: str, params: tuple = None) -> int:
        """执行更新操作（INSERT/UPDATE/DELETE）"""
        with self.get_cursor() as cursor:
            logger.info(f"执行SQL: {sql}")
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            affected_rows = cursor.rowcount
            logger.info(f"影响 {affected_rows} 行")
            return affected_rows

    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                logger.info(f"数据库连接测试成功: {result}")
                return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False

    def create_chat_record_table(self):
        """创建聊天记录表"""
        sql = """
        CREATE TABLE IF NOT EXISTS AI_CHAT_RECORD (
            CHAT_ID VARCHAR(100) PRIMARY KEY,
            USER_ID VARCHAR(50),
            SESSION_ID VARCHAR(100),
            AI_MODEL VARCHAR(50),
            ROLE VARCHAR(20),
            CONTENT CLOB,
            USER_PROMPT CLOB,
            SYSTEM_PROMPT CLOB,
            CREATE_TIME BIGINT,
            PROMPT_TOKENS INT,
            COMPLETION_TOKENS INT,
            TOTAL_TOKENS INT
        )
        """
        try:
            self.execute_update(sql)
            logger.info("✅ 聊天记录表创建成功")
        except Exception as e:
            logger.warning(f"创建表失败（可能已存在）: {e}")

    def insert_chat_record(self, record: Dict[str, Any]) -> bool:
        """插入聊天记录"""
        sql = """
        INSERT INTO AI_CHAT_RECORD (
            CHAT_ID, USER_ID, SESSION_ID, AI_MODEL, ROLE, CONTENT,
            USER_PROMPT, SYSTEM_PROMPT, CREATE_TIME,
            PROMPT_TOKENS, COMPLETION_TOKENS, TOTAL_TOKENS
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            record.get('chat_id'),
            record.get('user_id', ''),
            record.get('session_id'),
            record.get('ai_model'),
            record.get('role'),
            record.get('content'),
            record.get('user_prompt'),
            record.get('system_prompt'),
            record.get('create_time'),
            record.get('prompt_tokens', 0),
            record.get('completion_tokens', 0),
            record.get('total_tokens', 0)
        )

        try:
            self.execute_update(sql, params)
            return True
        except Exception as e:
            logger.error(f"插入聊天记录失败: {e}")
            return False

    def get_chat_records(self, session_id: str, limit: int = 30, offset: int = 0) -> List[Dict[str, Any]]:
        """获取聊天记录"""
        sql = """
        SELECT * FROM AI_CHAT_RECORD
        WHERE SESSION_ID = ?
        ORDER BY CREATE_TIME DESC
        LIMIT ? OFFSET ?
        """
        try:
            return self.execute_query(sql, (session_id, limit, offset))
        except Exception as e:
            logger.error(f"查询聊天记录失败: {e}")
            return []


# 全局数据库实例
db = DMDatabase()


# 初始化数据库
def init_database():
    """初始化数据库"""
    try:
        logger.info("=" * 60)
        logger.info("初始化达梦数据库")
        logger.info("=" * 60)

        # 测试连接
        if db.test_connection():
            logger.info("✅ 数据库连接正常")

            # 创建表
            db.create_chat_record_table()

            logger.info("✅ 数据库初始化完成")
            logger.info("=" * 60)
            return True
        else:
            logger.error("❌ 数据库连接失败")
            return False

    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        return False
