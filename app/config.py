import json
from pathlib import Path


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def server(self) -> dict:
        return self._config.get('server', {})

    @property
    def database(self) -> dict:
        return self._config.get('database', {})

    @property
    def ai(self) -> dict:
        return self._config.get('ai', {})

    @property
    def host(self) -> str:
        return self.server.get('host', '0.0.0.0')

    @property
    def port(self) -> int:
        return self.server.get('port', 9527)

    @property
    def base_path(self) -> str:
        return self.server.get('base_path', '/api/v1')

    @property
    def upload_path(self) -> str:
        return self.server.get('upload_path', 'upload')


# 全局配置实例
config = Config()
