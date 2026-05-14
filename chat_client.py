"""
Agent 交互式对话客户端
用于测试 Agent 服务的命令行工具
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime


class AgentClient:
    """Agent 客户端"""

    def __init__(self, url: str, session_id: str = "test-session"):
        self.url = url
        self.session_id = session_id
        self.ws = None
        self.connected = False
        self.ai_model = "glm-4"
        self.waiting_response = False  # 是否正在等待 AI 回复

    async def connect(self):
        """连接到服务器"""
        try:
            full_url = f"{self.url}?session_id={self.session_id}"
            print(f"\n🔗 正在连接到: {full_url}")
            self.ws = await websockets.connect(full_url)
            self.connected = True
            print("✅ 连接成功！\n")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    async def disconnect(self):
        """断开连接"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            print("\n👋 已断开连接")

    async def send_message(self, content: str, message_type: str = "aiMessage"):
        """发送消息"""
        if not self.connected:
            print("❌ 未连接到服务器")
            return

        message = {
            "type": message_type,
            "data": {
                "role": "user",
                "aiModel": self.ai_model if message_type == "aiMessage" else "",
                "content": [
                    {
                        "type": "text",
                        "content": content
                    }
                ]
            },
            "done": False
        }

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            print(f"❌ 发送失败: {e}")

    async def receive_messages(self):
        """接收消息"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("\n⚠️  连接已关闭")
            self.connected = False
        except Exception as e:
            print(f"\n❌ 接收消息错误: {e}")
            self.connected = False

    async def handle_message(self, data: dict):
        """处理接收到的消息"""
        msg_type = data.get("type")
        msg_data = data.get("data", {})
        done = data.get("done", False)

        if msg_type == "progress":
            # 进度节点：单独一行显示，带清行效果
            content_list = msg_data.get("content", [])
            if content_list:
                text = content_list[0].get("content", "")
                print(f"\r\033[K⏳ {text}", flush=True)

        elif msg_type == "aiMessage":
            content_list = msg_data.get("content", [])
            if content_list:
                content = content_list[0].get("content", "")
                if content:
                    print(content, end="", flush=True)

            if done:
                print("\n")  # 完成后换行
                self.waiting_response = False  # 解锁输入

        elif msg_type == "tts":
            content_list = msg_data.get("content", [])
            if content_list:
                audio_data = content_list[0].get("content", "")
                print(f"\n🔊 收到音频数据，长度: {len(audio_data)}")

    async def send_heartbeat(self):
        """发送心跳，保持 WebSocket 连接"""
        while self.connected:
            try:
                heartbeat = {
                    "type": "heartbeat",
                    "data": {
                        "role": "",
                        "content": [],
                        "aiModel": ""
                    },
                    "done": False
                }
                await self.ws.send(json.dumps(heartbeat))
                await asyncio.sleep(10)  # 每 10 秒一次，远小于 AI 处理时间
            except Exception:
                # 心跳失败说明连接已断，尝试重连
                if self.connected:
                    await self._reconnect()
                break

    async def _reconnect(self):
        """断线重连"""
        self.connected = False
        print("\n⚠️  连接断开，正在重连...")
        for i in range(3):
            await asyncio.sleep(2)
            try:
                full_url = f"{self.url}?session_id={self.session_id}"
                self.ws = await websockets.connect(full_url)
                self.connected = True
                # 重启心跳和接收任务
                asyncio.create_task(self.send_heartbeat())
                asyncio.create_task(self.receive_messages())
                print("✅ 重连成功\n")
                return
            except Exception:
                print(f"  重连失败（{i+1}/3）...")
        print("❌ 重连失败，请手动重启客户端\n")

    def print_help(self):
        """打印帮助信息"""
        print("\n" + "=" * 60)
        print("📖 命令帮助")
        print("=" * 60)
        print("/help          - 显示此帮助信息")
        print("/quit 或 /exit - 退出程序")
        print("/tts <文本>    - 语音播报")
        print("/model <模型>  - 切换 AI 模型 (glm-4, gpt-4, gpt-3.5-turbo)")
        print("/clear         - 清屏")
        print("/status        - 显示连接状态")
        print("\n直接输入消息即可与 AI 对话")
        print("=" * 60 + "\n")

    def print_status(self):
        """打印状态信息"""
        status = "✅ 已连接" if self.connected else "❌ 未连接"
        print(f"\n📊 状态: {status}")
        print(f"🔗 URL: {self.url}")
        print(f"🆔 Session: {self.session_id}")
        print(f"🤖 模型: {self.ai_model}\n")

    async def run(self):
        """运行客户端"""
        # 连接到服务器
        if not await self.connect():
            return

        # 启动心跳任务
        heartbeat_task = asyncio.create_task(self.send_heartbeat())

        # 启动接收消息任务
        receive_task = asyncio.create_task(self.receive_messages())

        # 打印欢迎信息
        print("=" * 60)
        print("🤖 Agent 交互式对话客户端")
        print("=" * 60)
        print("输入 /help 查看帮助，输入 /quit 退出")
        print("=" * 60 + "\n")

        try:
            while self.connected:
                # 等待 AI 回复期间不接受输入，每 0.2s 检查一次
                if self.waiting_response:
                    await asyncio.sleep(0.2)
                    continue

                # 获取用户输入
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "👤 你: "
                    )
                except EOFError:
                    break

                if not user_input.strip():
                    continue

                # 如果在等待输入期间 AI 开始回复（极小概率竞态），丢弃本次输入
                if self.waiting_response:
                    print("⏳ AI 正在回复中，请稍候...\n")
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    command = user_input.split()[0].lower()

                    if command in ["/quit", "/exit"]:
                        print("\n👋 再见！")
                        break

                    elif command == "/help":
                        self.print_help()

                    elif command == "/status":
                        self.print_status()

                    elif command == "/clear":
                        import os
                        os.system('cls' if os.name == 'nt' else 'clear')

                    elif command == "/model":
                        parts = user_input.split(maxsplit=1)
                        if len(parts) > 1:
                            self.ai_model = parts[1]
                            print(f"✅ 已切换到模型: {self.ai_model}\n")
                        else:
                            print("❌ 用法: /model <模型名称>\n")

                    elif command == "/tts":
                        parts = user_input.split(maxsplit=1)
                        if len(parts) > 1:
                            text = parts[1]
                            print(f"🔊 正在播报: {text}")
                            await self.send_message(text, "tts")
                        else:
                            print("❌ 用法: /tts <文本>\n")

                    else:
                        print(f"❌ 未知命令: {command}")
                        print("输入 /help 查看帮助\n")

                else:
                    # 发送普通消息，设置等待锁
                    self.waiting_response = True
                    await self.send_message(user_input)
                    print("🤖 AI: ", end="", flush=True)

        except KeyboardInterrupt:
            print("\n\n⚠️  收到中断信号")

        finally:
            # 清理
            heartbeat_task.cancel()
            receive_task.cancel()
            await self.disconnect()


async def main():
    """主函数"""
    # 默认配置
    url = "ws://localhost:9527/api/v1/agent/chat"
    session_id = f"cli-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 解析命令行参数
    if len(sys.argv) > 1:
        url = sys.argv[1]
    if len(sys.argv) > 2:
        session_id = sys.argv[2]

    # 创建并运行客户端
    client = AgentClient(url, session_id)
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
