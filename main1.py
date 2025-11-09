import asyncio
import json
import os
import uuid
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from implements.RagCore import RAGRetriever, RAGConfig
from implements.aiChatManager import (
    GeminiChatManager,
    OpenAIChatManager,
    SessionManager
)

# ============ 配置加载 ============
from config import AI_CONFIG, RAG_CONFIG, SERVER_CONFIG

# ============ FastAPI 应用初始化 ============
app = FastAPI(title="RAG API", version="2.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 全局实例 ============
rag_retriever: Optional[RAGRetriever] = None
gemini_manager: Optional[GeminiChatManager] = None
openai_manager: Optional[OpenAIChatManager] = None
session_manager = SessionManager()


def init_rag_retriever():
    """初始化RAG检索器"""
    global rag_retriever
    if rag_retriever is None:
        config = RAGConfig(**RAG_CONFIG)
        rag_retriever = RAGRetriever(config=config)
    return rag_retriever


def get_chat_manager(provider: str):
    """获取AI对话管理器"""
    global gemini_manager, openai_manager

    if provider == 'gemini':
        if gemini_manager is None:
            if not AI_CONFIG['gemini']['enabled']:
                raise HTTPException(status_code=400, detail="Gemini provider is not enabled")
            gemini_manager = GeminiChatManager(
                api_key=AI_CONFIG['gemini']['api_key'],
                model_name=AI_CONFIG['gemini']['model_name']
            )
        return gemini_manager

    elif provider == 'openai':
        if openai_manager is None:
            if not AI_CONFIG['openai']['enabled']:
                raise HTTPException(status_code=400, detail="OpenAI provider is not enabled")
            openai_manager = OpenAIChatManager(
                api_key=AI_CONFIG['openai']['api_key'],
                model_name=AI_CONFIG['openai']['model_name'],
                base_url=AI_CONFIG['openai'].get('base_url')
            )
        return openai_manager

    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


# ============ Pydantic Models ============

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    provider: str = 'gemini'  # 'gemini' 或 'openai'
    use_query_rewriting: bool = True
    n_results: Optional[int] = None


class SessionCreateRequest(BaseModel):
    provider: str = 'gemini'


class SessionRequest(BaseModel):
    session_id: str


class ReloadRequest(BaseModel):
    force_reload: bool = False


class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = None


# ============ API 端点 ============


def startup_event():
    """应用启动时初始化"""
    print("\n" + "=" * 70)
    print("🚀 启动 RAG API 服务...")
    print("=" * 70)

    # 初始化RAG检索器
    print("\n📦 初始化 RAG 检索器...")
    init_rag_retriever()

    # 预加载AI管理器（如果配置启用）
    if AI_CONFIG['gemini']['enabled']:
        print("\n📦 初始化 Gemini 管理器...")
        get_chat_manager('gemini')

    if AI_CONFIG['openai']['enabled']:
        print("\n📦 初始化 OpenAI 管理器...")
        get_chat_manager('openai')

    print("\n✅ 所有组件初始化完成")
startup_event()

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "message": "RAG API is running",
        "providers": {
            "gemini": AI_CONFIG['gemini']['enabled'],
            "openai": AI_CONFIG['openai']['enabled']
        }
    }


@app.get("/api/info")
async def get_info():
    """获取系统信息"""
    try:
        retriever = init_rag_retriever()
        stats = retriever.get_stats()

        return {
            'success': True,
            **stats,
            'active_sessions': len(session_manager.sessions),
            'config': {
                'max_results': retriever.config.max_results,
                'similarity_threshold': retriever.config.similarity_threshold,
                'use_hybrid_search': retriever.config.use_hybrid_search,
                'keyword_boost': retriever.config.keyword_boost,
                'context_window_size': retriever.config.context_window_size
            },
            'providers': {
                'gemini': {
                    'enabled': AI_CONFIG['gemini']['enabled'],
                    'model': AI_CONFIG['gemini']['model_name']
                },
                'openai': {
                    'enabled': AI_CONFIG['openai']['enabled'],
                    'model': AI_CONFIG['openai']['model_name']
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/create")
async def create_session(request: SessionCreateRequest):
    """创建新的会话"""
    # 验证provider
    if request.provider not in ['gemini', 'openai']:
        raise HTTPException(status_code=400, detail="Invalid provider. Must be 'gemini' or 'openai'")

    # 检查provider是否启用
    if not AI_CONFIG[request.provider]['enabled']:
        raise HTTPException(status_code=400, detail=f"{request.provider} provider is not enabled")

    session_id = str(uuid.uuid4())
    session_manager.create_session(session_id, provider=request.provider)

    return {
        'success': True,
        'session_id': session_id,
        'provider': request.provider,
        'message': f'会话创建成功 (使用 {request.provider.upper()} 模型)'
    }


@app.post("/api/session/clear")
async def clear_session(request: SessionRequest):
    """清空会话历史"""
    session_id = request.session_id
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail='会话不存在')

    session_manager.clear_session(session_id)
    return {
        'success': True,
        'message': '会话历史已清空'
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail='会话不存在')

    session_manager.delete_session(session_id)
    return {
        'success': True,
        'message': '会话已删除'
    }


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail='会话不存在')

    return {
        'success': True,
        'session_id': session_id,
        'provider': session['provider'],
        'history': session['history']
    }


@app.post("/api/ask/stream")
async def ask_question_stream(request: QueryRequest):
    """RAG 问答接口（流式传输）"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail='问题不能为空')

        session_id = request.session_id
        provider = request.provider

        # 验证provider
        if provider not in ['gemini', 'openai']:
            raise HTTPException(status_code=400, detail="Invalid provider")

        # 如果没有 session_id，创建新会话
        if not session_id:
            session_id = str(uuid.uuid4())
            session_manager.create_session(session_id, provider=provider)
        elif session_id not in session_manager.sessions:
            session_manager.create_session(session_id, provider=provider)
        else:
            # 验证会话的provider是否匹配
            session_provider = session_manager.get_provider(session_id)
            if session_provider != provider:
                raise HTTPException(
                    status_code=400,
                    detail=f"Session was created with {session_provider}, cannot use {provider}"
                )

        # 获取组件
        retriever = init_rag_retriever()
        chat_manager = get_chat_manager(provider)
        chat_history = session_manager.get_history(session_id)

        async def event_generator():
            try:
                # 发送 session_id
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'provider': provider}, ensure_ascii=False)}\n\n"

                # 查询重写（如果启用）
                search_query = query
                if request.use_query_rewriting:
                    search_query = chat_manager.rewrite_query(query)
                    yield f"data: {json.dumps({'type': 'rewrite', 'content': search_query}, ensure_ascii=False)}\n\n"

                # 检索相关文档
                search_results = retriever.search(search_query, n_results=request.n_results)

                if not search_results['documents'][0]:
                    yield f"data: {json.dumps({'type': 'error', 'content': '没有找到相关文档,无法回答问题。'}, ensure_ascii=False)}\n\n"
                    return

                # 扩展上下文窗口
                context_items = retriever.expand_context_with_window(search_results)

                # 发送检索到的源信息
                sources_info = [
                    {
                        'source': item['meta']['source'],
                        'section_path': item['meta']['section_path'],
                        'is_hit': item['is_hit']
                    }
                    for item in context_items
                ]
                yield f"data: {json.dumps({'type': 'sources', 'content': sources_info, 'count': len(sources_info)}, ensure_ascii=False)}\n\n"

                # 构建RAG提示词
                prompt = chat_manager.build_rag_prompt(query, context_items)

                # 流式生成答案
                full_answer = ""
                for chunk_data in chat_manager.generate_answer_stream(prompt, chat_history):
                    yield f"data: {chunk_data}\n\n"

                    # 收集完整答案
                    chunk_obj = json.loads(chunk_data)
                    if chunk_obj['type'] == 'content':
                        full_answer += chunk_obj['content']

                    await asyncio.sleep(0.01)

                # 更新会话历史
                session_manager.add_message(session_id, 'user', query)
                #print(type(chat_manager))
                if isinstance(chat_manager, OpenAIChatManager):
                    session_manager.add_message(session_id, 'assistant', full_answer)
                else:
                    session_manager.add_message(session_id, 'model', full_answer)


            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'处理请求时出错: {str(e)}'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    """RAG 问答接口（非流式）"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail='问题不能为空')

        session_id = request.session_id
        provider = request.provider

        # 验证并创建会话
        if not session_id:
            session_id = str(uuid.uuid4())
            session_manager.create_session(session_id, provider=provider)
        elif session_id not in session_manager.sessions:
            session_manager.create_session(session_id, provider=provider)

        # 获取组件
        retriever = init_rag_retriever()
        chat_manager = get_chat_manager(provider)
        chat_history = session_manager.get_history(session_id)

        # 查询重写
        search_query = query
        if request.use_query_rewriting:
            search_query = chat_manager.rewrite_query(query)

        # 检索
        search_results = retriever.search(search_query, n_results=request.n_results)

        if not search_results['documents'][0]:
            return {
                'success': False,
                'message': '没有找到相关文档',
                'session_id': session_id
            }

        # 扩展上下文
        context_items = retriever.expand_context_with_window(search_results)

        # 生成答案
        prompt = chat_manager.build_rag_prompt(query, context_items)
        answer = chat_manager.generate_answer(prompt, chat_history)

        # 更新历史
        session_manager.add_message(session_id, 'user', query)
        if isinstance(chat_manager, OpenAIChatManager):
            session_manager.add_message(session_id, 'assistant', answer)
        else:
            session_manager.add_message(session_id, 'model', answer)

        return {
            'success': True,
            'session_id': session_id,
            'provider': provider,
            'answer': answer,
            'sources': [
                {
                    'source': item['meta']['source'],
                    'section_path': item['meta']['section_path']
                }
                for item in context_items
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_documents(request: SearchRequest):
    """搜索相关文档（不生成答案）"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail='查询内容不能为空')

        retriever = init_rag_retriever()
        search_results = retriever.search(query, n_results=request.n_results)

        results = []
        if search_results['documents'][0]:
            for doc, metadata, distance in zip(
                    search_results['documents'][0],
                    search_results['metadatas'][0],
                    search_results['distances'][0]
            ):
                results.append({
                    'document': doc,
                    'source': metadata.get('source', ''),
                    'section_path': metadata.get('section_path', ''),
                    'keywords': metadata.get('keywords', ''),
                    'semantic_similarity': round((1 - distance) * 100, 2)
                })

        return {
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload")
async def reload_documents(request: ReloadRequest):
    """重新加载文档"""
    try:
        retriever = init_rag_retriever()
        retriever.load_documents_from_folder(
            folder_path="./docs",
            force_reload=request.force_reload
        )

        stats = retriever.get_stats()

        return {
            'success': True,
            'message': '文档重新加载完成',
            **stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 挂载静态文件（Web UI）
app.mount("/", StaticFiles(directory="web", html=True), name="web")


def start_server():
    """启动服务器"""
    import uvicorn

    print("\n✅ 系统初始化完成")
    print(f"🌐 Web UI 访问地址: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"📚 API 文档 (Swagger): http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}/docs")
    print("\n📡 可用的 API 端点:")
    print(f"   • GET  /api/health              - 健康检查")
    print(f"   • GET  /api/info                - 系统信息")
    print(f"   • POST /api/session/create      - 创建会话")
    print(f"   • POST /api/session/clear       - 清空会话")
    print(f"   • DELETE /api/session/:id       - 删除会话")
    print(f"   • GET  /api/session/:id/history - 获取历史")
    print(f"   • POST /api/ask                 - RAG 问答（非流式）")
    print(f"   • POST /api/ask/stream          - RAG 问答（流式）")
    print(f"   • POST /api/search              - 搜索文档")
    print(f"   • POST /api/reload              - 重新加载文档")
    print("=" * 70 + "\n")

    uvicorn.run(
        app,
        host=SERVER_CONFIG['host'],
        port=SERVER_CONFIG['port']
    )


if __name__ == "__main__":
    start_server()