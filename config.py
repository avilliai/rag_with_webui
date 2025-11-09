import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类 - 从YAML文件加载配置"""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config_data = self._load_config()
        self._apply_proxy_settings()

    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_file.exists():
            print(f"⚠️  配置文件 {self.config_file} 不存在，使用默认配置")
            return self._get_default_config()

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print(f"✅ 成功加载配置文件: {self.config_file}")
                return config
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            print("使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'ai': {
                'gemini': {
                    'enabled': True,
                    'api_key': '',
                    'model_name': 'gemini-2.5-flash'
                },
                'openai': {
                    'enabled': False,
                    'api_key': '',
                    'model_name': 'gpt-4-turbo-preview',
                    'base_url': None
                }
            },
            'rag': {
                'max_results': 5,
                'similarity_threshold': 0.4,
                'use_hybrid_search': True,
                'keyword_boost': 0.4,
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'context_window_size': 3
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000
            },
            'proxy': {
                'enabled': False,
                'http_proxy': 'http://127.0.0.1:10809',
                'https_proxy': 'http://127.0.0.1:10809'
            }
        }

    def _apply_proxy_settings(self):
        """应用代理设置到环境变量"""
        proxy_config = self.config_data.get('proxy', {})

        if proxy_config.get('enabled', False):
            http_proxy = proxy_config.get('http_proxy')
            https_proxy = proxy_config.get('https_proxy')

            if http_proxy:
                os.environ["http_proxy"] = http_proxy
                print(f"✅ 设置 HTTP 代理: {http_proxy}")

            if https_proxy:
                os.environ["https_proxy"] = https_proxy
                print(f"✅ 设置 HTTPS 代理: {https_proxy}")
        else:
            # 清除代理设置（如果之前设置过）
            if "http_proxy" in os.environ:
                del os.environ["http_proxy"]
            if "https_proxy" in os.environ:
                del os.environ["https_proxy"]

    def _get_env_or_config(self, env_key: str, config_value: Any) -> Any:
        """优先使用环境变量，否则使用配置文件的值"""
        env_value = os.getenv(env_key)
        return env_value if env_value is not None else config_value

    @property
    def ai_config(self) -> Dict[str, Any]:
        """获取AI配置（支持环境变量覆盖）"""
        ai_cfg = self.config_data.get('ai', {})

        # Gemini配置
        gemini_cfg = ai_cfg.get('gemini', {})
        gemini_api_key = self._get_env_or_config('GEMINI_API_KEY', gemini_cfg.get('api_key', ''))

        # OpenAI配置
        openai_cfg = ai_cfg.get('openai', {})
        openai_api_key = self._get_env_or_config('OPENAI_API_KEY', openai_cfg.get('api_key', ''))
        openai_base_url = self._get_env_or_config('OPENAI_BASE_URL', openai_cfg.get('base_url'))

        return {
            'gemini': {
                'enabled': gemini_cfg.get('enabled', True),
                'api_key': gemini_api_key,
                'model_name': gemini_cfg.get('model_name', 'gemini-2.5-flash')
            },
            'openai': {
                'enabled': openai_cfg.get('enabled', False),
                'api_key': openai_api_key,
                'model_name': openai_cfg.get('model_name', 'gpt-4-turbo-preview'),
                'base_url': openai_base_url
            }
        }

    @property
    def rag_config(self) -> Dict[str, Any]:
        """获取RAG配置"""
        return self.config_data.get('rag', {
            'max_results': 5,
            'similarity_threshold': 0.4,
            'use_hybrid_search': True,
            'keyword_boost': 0.4,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'context_window_size': 3
        })

    @property
    def server_config(self) -> Dict[str, Any]:
        """获取服务器配置（支持环境变量覆盖）"""
        server_cfg = self.config_data.get('server', {})

        host = self._get_env_or_config('SERVER_HOST', server_cfg.get('host', '0.0.0.0'))
        port = self._get_env_or_config('SERVER_PORT', server_cfg.get('port', 8000))

        # 确保port是整数
        if isinstance(port, str):
            port = int(port)

        return {
            'host': host,
            'port': port
        }

    @property
    def proxy_config(self) -> Dict[str, Any]:
        """获取代理配置"""
        return self.config_data.get('proxy', {
            'enabled': False,
            'http_proxy': 'http://127.0.0.1:10809',
            'https_proxy': 'http://127.0.0.1:10809'
        })

    def reload(self):
        """重新加载配置文件"""
        print("\n🔄 重新加载配置文件...")
        self.config_data = self._load_config()
        self._apply_proxy_settings()
        print("✅ 配置重新加载完成")


# 创建全局配置实例
config = Config()

# 导出配置（保持向后兼容）
AI_CONFIG = config.ai_config
RAG_CONFIG = config.rag_config
SERVER_CONFIG = config.server_config
PROXY_CONFIG = config.proxy_config


def reload_config():
    """重新加载配置的便捷函数"""
    global AI_CONFIG, RAG_CONFIG, SERVER_CONFIG, PROXY_CONFIG
    config.reload()
    AI_CONFIG = config.ai_config
    RAG_CONFIG = config.rag_config
    SERVER_CONFIG = config.server_config
    PROXY_CONFIG = config.proxy_config