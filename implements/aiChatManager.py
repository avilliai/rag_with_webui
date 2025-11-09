from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator
import json


class BaseChatManager(ABC):
    """AI对话管理器抽象基类"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def rewrite_query(self, query: str) -> str:
        """重写查询以优化检索"""
        pass

    @abstractmethod
    def generate_answer(self, prompt: str, chat_history: Optional[List[Dict]] = None) -> str:
        """生成答案（非流式）"""
        pass

    @abstractmethod
    def generate_answer_stream(
            self,
            prompt: str,
            chat_history: Optional[List[Dict]] = None
    ) -> Generator[str, None, None]:
        """生成答案（流式）"""
        pass

    def build_rag_prompt(self, query: str, context_items: List[Dict]) -> str:
        """构建RAG提示词"""
        context_parts = []
        print("\n📚 构建最终上下文:")
        for item in context_items:
            meta = item['meta']
            source_info = f"[来源: {meta.get('source', '未知')} | 章节: {meta.get('section_path', 'N/A')}]"
            hit_marker = "🎯" if item.get('is_hit') else "📄"
            print(f"   {hit_marker} {source_info}")
            context_parts.append(f"{source_info}\n{item['doc']}")

        context = "\n\n---\n\n".join(context_parts)
        print(f"\n🔍 最终上下文包含 {len(context_parts)} 个文档块, 总长度 {len(context)} 字符")

        return f"""你是一个专业的政治学知识解答模型。请严格基于以下检索到的文档内容，以系统、学术化的方式回答用户的问题。
- 综合所有提供的信息，给出全面而有条理的答案。
- 如果文档内容不足以回答，请明确指出。
- 以Markdown格式进行回复。

--- [检索到的文档] ---
{context}
--- [检索到的文档结束] ---

问题: {query}

请提供详细且准确的答案:"""


class GeminiChatManager(BaseChatManager):
    """Gemini AI对话管理器"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        super().__init__(model_name)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.chat_model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini 模型初始化完成: {model_name}")

    def rewrite_query(self, query: str) -> str:
        """使用Gemini重写查询"""
        print(f"\n🔄 正在重写查询 (Gemini)...")
        prompt = f"""你是一名检索优化专家。请将以下用户问题改写为一个信息更丰富的陈述句，用于向量数据库的语义检索。请专注于核心意图，补充可能的上下文，使其更像一个"答案"的片段。
直接返回改写后的文本，不要包含任何解释或前缀。

原始问题: "{query}"

改写后的检索查询:
"""
        try:
            response = self.chat_model.generate_content(prompt)
            rewritten_query = response.text.strip().replace("*", "")
            print(f"   - 原始查询: {query}")
            print(f"   - 重写后: {rewritten_query}")
            return rewritten_query
        except Exception as e:
            print(f"⚠️ 查询重写失败: {e}，将使用原始查询。")
            return query

    def generate_answer(self, prompt: str, chat_history: Optional[List[Dict]] = None) -> str:
        """使用Gemini生成答案（非流式）"""
        try:
            print("\n💡 正在生成答案 (Gemini)...")

            if chat_history:
                # 使用聊天会话
                chat = self.chat_model.start_chat(history=chat_history)
                response = chat.send_message(prompt)
            else:
                response = self.chat_model.generate_content(prompt)

            return response.text
        except Exception as e:
            return f"❌ 生成答案时出错: {e}"

    def generate_answer_stream(
            self,
            prompt: str,
            chat_history: Optional[List[Dict]] = None
    ) -> Generator[str, None, None]:
        """使用Gemini生成答案（流式）"""
        try:
            if chat_history:
                chat = self.chat_model.start_chat(history=chat_history)
                response = chat.send_message(prompt, stream=True)
            else:
                response = self.chat_model.generate_content(prompt, stream=True)

            for chunk in response:
                if chunk.text:
                    yield json.dumps({
                        'type': 'content',
                        'content': chunk.text
                    }, ensure_ascii=False) + '\n'

            yield json.dumps({
                'type': 'done',
                'content': ''
            }, ensure_ascii=False) + '\n'
        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'content': f'生成答案时出错: {str(e)}'
            }, ensure_ascii=False) + '\n'


class OpenAIChatManager(BaseChatManager):
    """OpenAI对话管理器"""

    def __init__(
            self,
            api_key: str,
            model_name: str = "gpt-4-turbo-preview",
            base_url: Optional[str] = None
    ):
        super().__init__(model_name)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url  # 支持自定义endpoint（如OpenAI兼容接口）
        )
        print(f"✅ OpenAI 模型初始化完成: {model_name}")

    def rewrite_query(self, query: str) -> str:
        """使用OpenAI重写查询"""
        print(f"\n🔄 正在重写查询 (OpenAI)...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名检索优化专家。将用户问题改写为信息指向更明确的简单陈述句，用于向量数据库的语义检索。请专注于核心意图。"
                    },
                    {
                        "role": "user",
                        "content": f'原始问题: "{query}"\n\n改写后的检索查询:'
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )

            rewritten_query = response.choices[0].message.content.strip()
            print(f"   - 原始查询: {query}")
            print(f"   - 重写后: {rewritten_query}")
            return rewritten_query
        except Exception as e:
            print(f"⚠️ 查询重写失败: {e}，将使用原始查询。")
            return query

    def generate_answer(self, prompt: str, chat_history: Optional[List[Dict]] = None) -> str:
        """使用OpenAI生成答案（非流式）"""
        try:
            print("\n💡 正在生成答案 (OpenAI)...")

            messages = []

            # 添加历史记录
            if chat_history:
                for msg in chat_history:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # 添加当前prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 生成答案时出错: {e}"

    def generate_answer_stream(
            self,
            prompt: str,
            chat_history: Optional[List[Dict]] = None
    ) -> Generator[str, None, None]:
        """使用OpenAI生成答案（流式）"""
        try:
            messages = []

            # 添加历史记录
            if chat_history:
                for msg in chat_history:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # 添加当前prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield json.dumps({
                        'type': 'content',
                        'content': chunk.choices[0].delta.content
                    }, ensure_ascii=False) + '\n'

            yield json.dumps({
                'type': 'done',
                'content': ''
            }, ensure_ascii=False) + '\n'
        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'content': f'生成答案时出错: {str(e)}'
            }, ensure_ascii=False) + '\n'


class SessionManager:
    """会话管理器 - 管理不同AI提供商的对话历史"""

    def __init__(self):
        # 格式: {session_id: {'provider': 'gemini'/'openai', 'history': [...]}}
        self.sessions: Dict[str, Dict] = {}

    def create_session(self, session_id: str, provider: str = 'gemini') -> None:
        """创建新会话"""
        self.sessions[session_id] = {
            'provider': provider,
            'history': []
        }

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        return self.sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到会话历史"""
        if session_id not in self.sessions:
            return

        provider = self.sessions[session_id]['provider']

        if provider == 'gemini':
            #print(role)
            # Gemini格式
            self.sessions[session_id]['history'].append({
                'role': role,
                'parts': [{'text': content}]
            })
        else:  # OpenAI格式
            self.sessions[session_id]['history'].append({
                'role': role,
                'content': content
            })

    def get_history(self, session_id: str) -> List[Dict]:
        """获取会话历史"""
        session = self.sessions.get(session_id)
        return session['history'] if session else []

    def clear_session(self, session_id: str) -> None:
        """清空会话历史"""
        if session_id in self.sessions:
            self.sessions[session_id]['history'] = []

    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_provider(self, session_id: str) -> Optional[str]:
        """获取会话使用的AI提供商"""
        session = self.sessions.get(session_id)
        return session['provider'] if session else None