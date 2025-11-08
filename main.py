"""
改进的 Gemini RAG 系统 - 混合检索策略
核心改进: 关键词匹配 + 语义检索 双重保障
专门优化人名、专有名词的检索准确度
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio
import json
import uuid
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import re
from collections import defaultdict

API_KEY = 'AIzaSyCNwmo17IETTpEAhCp9mvrtaovXteITZDM'

genai.configure(api_key=API_KEY)
os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"


class RAGConfig:
    """RAG 系统配置"""
    def __init__(
        self,
        max_results: int = 8,
        similarity_threshold: Optional[float] = None,
        chunk_size: int = 1000,  # 增大以保留更多上下文
        min_chunk_size: int = 200,
        chunk_overlap: int = 250,  # 增加重叠
        use_hybrid_search: bool = True,  # 启用混合检索
        keyword_boost: float = 0.3  # 关键词匹配的权重提升
    ):
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_search = use_hybrid_search
        self.keyword_boost = keyword_boost


class GeminiRAG:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        config: Optional[RAGConfig] = None
    ):
        """初始化 RAG 系统"""
        self.persist_directory = persist_directory
        self.config = config or RAGConfig()

        print(f"📦 初始化 ChromaDB (存储路径: {persist_directory})...")
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"✅ 发现已存在的数据库,包含 {existing_count} 个文档块")

        print("📦 加载嵌入模型...")
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ 嵌入模型加载完成")

        self.chat_model = genai.GenerativeModel('gemini-2.5-flash')

        print(f"\n⚙️  RAG 配置:")
        print(f"   • 检索策略: {'混合检索 (关键词+语义)' if self.config.use_hybrid_search else '纯语义检索'}")
        print(f"   • 最大检索数: {self.config.max_results}")
        print(f"   • 相似度阈值: {self.config.similarity_threshold or '自动'}")
        print(f"   • 分块大小: {self.config.chunk_size} 字符")
        print(f"   • 重叠大小: {self.config.chunk_overlap} 字符")
        if self.config.use_hybrid_search:
            print(f"   • 关键词权重: {self.config.keyword_boost}")

    def _get_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        embedding = self.embed_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件的 MD5 哈希值"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _read_markdown_file(self, file_path: Path) -> str:
        """读取 Markdown 文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  读取文件失败 {file_path}: {e}")
            return None

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        keywords = []

        # 提取 #标签
        hashtags = re.findall(r'#([^\s#]+)', text)
        keywords.extend(hashtags)

        # 提取加粗的文本
        bold_text = re.findall(r'\*\*([^*]+)\*\*', text)
        keywords.extend([t.strip() for t in bold_text if 3 < len(t.strip()) < 30])

        # 提取标题中的关键词
        headers = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        keywords.extend([h.strip() for h in headers if len(h.strip()) < 50])

        # 提取人名模式（中文姓名，通常2-4个字）
        # 匹配常见的学者名字模式
        names = re.findall(r'([·\u4e00-\u9fa5]{2,6}(?:的|、|与|和|及)?)', text)
        potential_names = [n.strip('的、与和及') for n in names
                          if 2 <= len(n.strip('的、与和及')) <= 6]
        keywords.extend(potential_names)

        return list(set(keywords))

    def _normalize_text(self, text: str) -> str:
        """文本归一化：去除多余空格、统一标点等"""
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 统一引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()

    def _extract_query_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        keywords = []

        # 提取可能的人名（2-6个汉字）
        names = re.findall(r'[\u4e00-\u9fa5·]{2,6}', query)
        keywords.extend(names)

        # 提取常见的查询关键词
        key_phrases = ['政治思想', '理论', '观点', '学说', '主张', '批判', '评价']
        for phrase in key_phrases:
            if phrase in query:
                keywords.append(phrase)

        return keywords

    def _keyword_match_score(self, query: str, doc_text: str, doc_keywords: str) -> float:
        """计算关键词匹配分数"""
        query_lower = query.lower()
        doc_text_lower = doc_text.lower()
        doc_keywords_lower = doc_keywords.lower() if doc_keywords else ""

        score = 0.0
        query_keywords = self._extract_query_keywords(query)

        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            # 精确匹配关键词
            if keyword_lower in doc_keywords_lower:
                score += 0.5  # 关键词字段匹配
            if keyword_lower in doc_text_lower:
                # 计算出现次数（上限为5次）
                count = min(doc_text_lower.count(keyword_lower), 5)
                score += count * 0.1  # 文本中匹配

        return min(score, 1.0)  # 限制在0-1之间

    def _parse_document_structure(self, text: str) -> List[Dict]:
        """解析文档结构"""
        lines = text.split('\n')
        sections = []
        current_section = {
            'content': [],
            'headers': [],
            'line_start': 0
        }
        header_stack = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # 检测标题
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line_stripped)
            if header_match:
                # 保存当前section
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    current_section['line_end'] = i
                    sections.append(current_section.copy())

                # 更新标题栈
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                while header_stack and header_stack[-1]['level'] >= level:
                    header_stack.pop()
                header_stack.append({'level': level, 'title': title})

                # 开始新section
                current_section = {
                    'content': [line],
                    'headers': [h['title'] for h in header_stack],
                    'header_level': level,
                    'line_start': i
                }
            elif line_stripped:
                current_section['content'].append(line)
            elif len(current_section['content']) > 5:  # 空行且有足够内容
                # 保存section
                current_section['content'] = '\n'.join(current_section['content'])
                current_section['line_end'] = i
                current_section['headers'] = [h['title'] for h in header_stack]
                sections.append(current_section.copy())

                # 开始新section
                current_section = {
                    'content': [],
                    'headers': [h['title'] for h in header_stack],
                    'line_start': i + 1
                }

        # 保存最后一个section
        if current_section['content']:
            if isinstance(current_section['content'], list):
                current_section['content'] = '\n'.join(current_section['content'])
            current_section['line_end'] = len(lines)
            current_section['headers'] = [h['title'] for h in header_stack]
            sections.append(current_section)

        return sections

    def _smart_chunk_document(self, text: str, file_name: str) -> List[tuple]:
        """智能分块文档"""
        sections = self._parse_document_structure(text)
        chunks = []
        chunk_idx = 0

        i = 0
        while i < len(sections):
            section = sections[i]

            # 构建标题上下文
            headers_text = ' > '.join(section.get('headers', []))
            header_prefix = f"# {headers_text}\n\n" if headers_text else ""

            # 收集内容
            chunk_content = []
            chunk_sections = []
            current_size = len(header_prefix)

            j = i
            while j < len(sections):
                candidate = sections[j]
                content = candidate['content']
                content_size = len(content)

                # 检查是否是新的主要主题（## 或 # 级别）
                if (j > i and
                    candidate.get('header_level') and
                    candidate['header_level'] <= 2):
                    break

                # 检查大小限制
                if current_size + content_size > self.config.chunk_size and chunk_content:
                    break

                chunk_content.append(content)
                chunk_sections.append(j)
                current_size += content_size + 2
                j += 1

            # 如果没有收集到内容（单个section过大）
            if not chunk_content and i < len(sections):
                content = sections[i]['content']
                # 分割长内容
                parts = self._split_long_text(content, self.config.chunk_size - len(header_prefix))
                for part_idx, part in enumerate(parts):
                    full_text = header_prefix + part
                    keywords = self._extract_keywords(full_text)

                    chunk_meta = {
                        'section_path': headers_text,
                        'keywords': ', '.join(keywords),
                        'section_indices': str(i),
                        'is_split': True,
                        'part': f"{part_idx + 1}/{len(parts)}"
                    }

                    chunks.append((full_text, f"{file_name}_chunk_{chunk_idx}",
                                 len(full_text), chunk_meta))
                    chunk_idx += 1
                i += 1
                continue

            # 构建完整chunk
            full_content = '\n\n'.join(chunk_content)
            full_text = header_prefix + full_content
            keywords = self._extract_keywords(full_text)

            chunk_meta = {
                'section_path': headers_text,
                'keywords': ', '.join(keywords),
                'section_indices': f"{min(chunk_sections)}-{max(chunk_sections)}" if len(chunk_sections) > 1 else str(chunk_sections[0]),
                'section_count': len(chunk_sections),
                'is_split': False
            }

            chunks.append((full_text, f"{file_name}_chunk_{chunk_idx}",
                         len(full_text), chunk_meta))
            chunk_idx += 1

            # 决定重叠策略
            if len(chunk_sections) > 2:
                # 从倒数第二个section开始重叠
                i = chunk_sections[-2]
            else:
                i = j

        return chunks

    def _split_long_text(self, text: str, max_size: int) -> List[str]:
        """分割超长文本"""
        if len(text) <= max_size:
            return [text]

        parts = []
        # 按句子分割
        sentences = re.split(r'([。！？\n]+)', text)

        current = ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""

            if len(current) + len(sentence) + len(delimiter) > max_size and current:
                parts.append(current)
                # 保留一些重叠
                overlap = current[-self.config.chunk_overlap:] if len(current) > self.config.chunk_overlap else current
                current = overlap + sentence + delimiter
            else:
                current += sentence + delimiter

        if current:
            parts.append(current)

        return parts if parts else [text[:max_size]]

    def load_documents_from_folder(self, folder_path: str = "./docs", force_reload: bool = False):
        """从文件夹递归加载所有 Markdown 文档"""
        docs_path = Path(folder_path)

        if not docs_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return

        md_files = list(docs_path.rglob("*.md"))

        if not md_files:
            print(f"⚠️  在 {folder_path} 中没有找到 Markdown 文件")
            return

        print(f"\n📁 找到 {len(md_files)} 个 Markdown 文件")

        existing_docs = self.collection.get()
        existing_ids = set(existing_docs['ids']) if existing_docs['ids'] else set()

        existing_hashes = {}
        if existing_docs['metadatas']:
            for metadata in existing_docs['metadatas']:
                source = metadata.get('source', '')
                file_hash = metadata.get('file_hash', '')
                if source and file_hash and source not in existing_hashes:
                    existing_hashes[source] = file_hash

        print(f"📋 数据库中已有 {len(existing_hashes)} 个不同的文档文件")

        new_docs = []
        new_ids = []
        new_metadatas = []
        updated_count = 0
        skipped_count = 0
        new_count = 0

        for md_file in md_files:
            relative_path = md_file.relative_to(docs_path)
            source_path = str(relative_path).replace('\\', '/')
            current_hash = self._get_file_hash(str(md_file))

            if source_path in existing_hashes:
                if not force_reload and existing_hashes[source_path] == current_hash:
                    skipped_count += 1
                    continue
                else:
                    print(f"🔄 检测到文件变更: {source_path}")
                    updated_count += 1
                    ids_to_delete = [eid for eid in existing_ids
                                    if eid.startswith(f"{source_path}_chunk_")]
                    if ids_to_delete:
                        self.collection.delete(ids=ids_to_delete)
            else:
                new_count += 1

            content = self._read_markdown_file(md_file)

            if content:
                chunks = self._smart_chunk_document(content, source_path)

                # 显示详细信息（前3个文件）
                if len(new_docs) < 50:
                    print(f"📄 {source_path}: {len(content)} 字符 → {len(chunks)} 块")

                for chunk_text, chunk_id, chunk_size, chunk_meta in chunks:
                    new_docs.append(chunk_text)
                    new_ids.append(chunk_id)

                    metadata = {
                        "source": source_path,
                        "file_name": md_file.name,
                        "file_hash": current_hash,
                        "file_size": md_file.stat().st_size,
                        "chunk_id": chunk_id,
                        "chunk_size": chunk_size,
                        **chunk_meta
                    }
                    new_metadatas.append(metadata)

        if new_docs:
            print(f"\n💾 正在添加 {len(new_docs)} 个文档块...")
            print(f"   📝 新增: {new_count} 个文件")
            if updated_count > 0:
                print(f"   🔄 更新: {updated_count} 个文件")

            print("🔄 生成嵌入向量（可能需要几分钟）...")
            # 批量生成嵌入以提高效率
            batch_size = 100
            embeddings = []
            for i in range(0, len(new_docs), batch_size):
                batch = new_docs[i:i+batch_size]
                batch_embeddings = [self._get_embedding(doc) for doc in batch]
                embeddings.extend(batch_embeddings)
                if len(new_docs) > batch_size:
                    print(f"   进度: {min(i+batch_size, len(new_docs))}/{len(new_docs)}")

            self.collection.add(
                documents=new_docs,
                embeddings=embeddings,
                ids=new_ids,
                metadatas=new_metadatas
            )

            print(f"✅ 成功添加 {len(new_docs)} 个文档块")
        else:
            print(f"\n✅ 所有文档已是最新")

        if skipped_count > 0:
            print(f"⏭️  跳过 {skipped_count} 个未修改的文档")

        print(f"\n📊 数据库共有 {self.collection.count()} 个文档块")

    def search(self, query: str, n_results: Optional[int] = None,
               similarity_threshold: Optional[float] = None) -> dict:
        """混合检索：关键词匹配 + 语义相似度"""
        n_results = n_results or self.config.max_results
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold

        # 语义检索
        query_embedding = self._get_embedding(query)
        search_n = min(n_results * 5, self.collection.count())  # 检索更多候选

        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_n
        )

        if not semantic_results or not semantic_results.get('documents') or not semantic_results['documents'][0]:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        # 如果启用混合检索，重新排序
        if self.config.use_hybrid_search:
            # 计算混合分数
            scored_results = []
            for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                semantic_results['ids'][0],
                semantic_results['documents'][0],
                semantic_results['metadatas'][0],
                semantic_results['distances'][0]
            )):
                # 语义分数（distance越小越好，转换为相似度）
                semantic_score = 1 - distance

                # 关键词匹配分数
                doc_keywords = metadata.get('keywords', '')
                keyword_score = self._keyword_match_score(query, doc, doc_keywords)

                # 混合分数
                final_score = semantic_score * (1 - self.config.keyword_boost) + \
                             keyword_score * self.config.keyword_boost

                scored_results.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': metadata,
                    'distance': distance,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'final_score': final_score
                })

            # 按最终分数排序
            scored_results.sort(key=lambda x: x['final_score'], reverse=True)

            # 应用阈值过滤（如果设置了）
            if similarity_threshold is not None:
                # 将阈值应用于semantic distance
                scored_results = [r for r in scored_results if r['distance'] <= similarity_threshold]

            # 限制结果数量
            scored_results = scored_results[:n_results]

            # 构建返回格式
            return {
                'ids': [[r['id'] for r in scored_results]],
                'documents': [[r['document'] for r in scored_results]],
                'metadatas': [[r['metadata'] for r in scored_results]],
                'distances': [[r['distance'] for r in scored_results]],
                'scores': [[r['final_score'] for r in scored_results]],  # 额外返回混合分数
                'keyword_scores': [[r['keyword_score'] for r in scored_results]]
            }

        # 非混合检索，直接返回语义结果
        if similarity_threshold is not None:
            filtered_indices = [i for i, dist in enumerate(semantic_results['distances'][0])
                              if dist <= similarity_threshold]
            filtered_indices = filtered_indices[:n_results]

            return {
                'ids': [[semantic_results['ids'][0][i] for i in filtered_indices]],
                'documents': [[semantic_results['documents'][0][i] for i in filtered_indices]],
                'metadatas': [[semantic_results['metadatas'][0][i] for i in filtered_indices]],
                'distances': [[semantic_results['distances'][0][i] for i in filtered_indices]]
            }

        # 限制数量
        if len(semantic_results['ids'][0]) > n_results:
            return {
                'ids': [semantic_results['ids'][0][:n_results]],
                'documents': [semantic_results['documents'][0][:n_results]],
                'metadatas': [semantic_results['metadatas'][0][:n_results]],
                'distances': [semantic_results['distances'][0][:n_results]]
            }

        return semantic_results

    def generate_answer_stream(self, query: str, chat_history: list = None,
                               n_results: Optional[int] = None,
                               similarity_threshold: Optional[float] = None):
        """RAG 流式生成答案（支持多轮对话）"""
        # 检索相关文档
        search_results = self.search(query, n_results, similarity_threshold)

        has_results = (
                search_results and
                search_results.get('documents') and
                len(search_results['documents']) > 0 and
                len(search_results['documents'][0]) > 0
        )

        if not has_results:
            yield json.dumps({
                'type': 'error',
                'content': '没有找到相关文档,无法回答问题。'
            }, ensure_ascii=False) + '\n'
            return

        # 构建上下文
        context_parts = []
        sources_info = []

        for i, (doc, metadata, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
        )):
            source = metadata.get('source', f'文档{i + 1}')
            section_path = metadata.get('section_path', '')
            keywords = metadata.get('keywords', '')
            semantic_score = 1 - distance
            keyword_score = search_results.get('keyword_scores', [[]])[0][
                i] if 'keyword_scores' in search_results else 0

            source_info = f"[来源: {source}"
            if section_path:
                source_info += f" | 章节: {section_path}"
            if keywords:
                kw_list = keywords.split(', ')[:5]
                source_info += f" | 关键词: {', '.join(kw_list)}"
            source_info += f" | 语义: {semantic_score:.2%}"
            if keyword_score > 0:
                source_info += f" | 关键词: {keyword_score:.2%}"
            source_info += "]"

            context_parts.append(f"{source_info}\n{doc}")

            sources_info.append({
                'source': source,
                'section_path': section_path,
                'keywords': keywords.split(', ')[:5] if keywords else [],
                'semantic_similarity': round(semantic_score * 100, 2),
                'keyword_score': round(keyword_score * 100, 2)
            })

        context = "\n\n---\n\n".join(context_parts)

        # 先发送检索到的文档信息
        yield json.dumps({
            'type': 'sources',
            'content': sources_info,
            'count': len(sources_info)
        }, ensure_ascii=False) + '\n'

        # 构建包含历史的 prompt
        system_prompt = """你是一个专业的政治学知识解答模型，你必须基于检索到的文档内容回答问题。
    给出系统、学术化的解答。你不被允许遗漏任何文档中的信息。
    如果文档中没有相关信息,请说明无法回答。"""

        # 创建或使用现有的 chat 会话
        if chat_history:
            # 使用 genai 的新接口创建 chat
            chat = self.chat_model.start_chat(history=chat_history)
        else:
            chat = self.chat_model.start_chat()

        # 构建用户消息
        user_message = f"""检索到的文档:
    {context}

    问题: {query}

    请提供详细且准确的答案:"""

        # 流式生成
        try:
            response = chat.send_message(user_message, stream=True)

            for chunk in response:
                if chunk.text:
                    yield json.dumps({
                        'type': 'content',
                        'content': chunk.text
                    }, ensure_ascii=False) + '\n'

            # 发送完成信号
            yield json.dumps({
                'type': 'done',
                'content': ''
            }, ensure_ascii=False) + '\n'

        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'content': f'生成答案时出错: {str(e)}'
            }, ensure_ascii=False) + '\n'
    def generate_answer(self, query: str, n_results: Optional[int] = None,
                       similarity_threshold: Optional[float] = None) -> str:
        """RAG: 检索相关文档并生成答案"""
        search_results = self.search(query, n_results, similarity_threshold)

        has_results = (
            search_results and
            search_results.get('documents') and
            len(search_results['documents']) > 0 and
            len(search_results['documents'][0]) > 0
        )

        if not has_results:
            return "❌ 没有找到相关文档,无法回答问题。"

        # 构建上下文
        context_parts = []
        for i, (doc, metadata, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
        )):
            source = metadata.get('source', f'文档{i + 1}')
            section_path = metadata.get('section_path', '')
            keywords = metadata.get('keywords', '')
            semantic_score = 1 - distance

            # 如果有混合分数，也显示
            keyword_score = search_results.get('keyword_scores', [[]])[0][i] if 'keyword_scores' in search_results else 0

            source_info = f"[来源: {source}"
            if section_path:
                source_info += f" | 章节: {section_path}"
            if keywords:
                kw_list = keywords.split(', ')[:5]
                source_info += f" | 关键词: {', '.join(kw_list)}"
            source_info += f" | 语义: {semantic_score:.2%}"
            if keyword_score > 0:
                source_info += f" | 关键词: {keyword_score:.2%}"
            source_info += "]"

            context_parts.append(f"{source_info}\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

        print(f"\n🔍 检索到 {len(context_parts)} 个相关文档块")

        prompt = f"""你是一个专业的政治学知识解答模型，你必须基于以下检索到的文档内容回答问题。给出系统、学术化的解答。你不被允许遗漏任何文档中的信息。如果文档中没有相关信息,请说明无法回答。

检索到的文档:
{context}

问题: {query}

请提供详细且准确的答案:"""

        try:
            response = self.chat_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"生成答案时出错: {e}"

    def get_collection_info(self):
        """获取集合信息"""
        count = self.collection.count()
        return f"集合中共有 {count} 个文档块"

    def list_documents(self, show_sample: bool = False):
        """列出所有文档"""
        results = self.collection.get()
        print(f"\n📚 数据库中的文档列表:")
        print("=" * 80)

        docs_by_source = defaultdict(list)
        for doc_id, metadata in zip(results['ids'], results['metadatas']):
            source = metadata.get('source', doc_id)
            docs_by_source[source].append(metadata)

        for i, (source, metadatas) in enumerate(sorted(docs_by_source.items()), 1):
            file_size = metadatas[0].get('file_size', 0)
            chunk_count = len(metadatas)
            total_chunk_size = sum(m.get('chunk_size', 0) for m in metadatas)
            avg_chunk_size = total_chunk_size // chunk_count if chunk_count > 0 else 0

            print(f"\n{i}. {source}")
            print(f"   大小: {file_size:,} bytes → {chunk_count} 块 (平均 {avg_chunk_size} 字符/块)")

            if show_sample and i <= 3:
                # 显示前3个块的关键词
                for j, meta in enumerate(metadatas[:3]):
                    keywords = meta.get('keywords', '')
                    if keywords:
                        kw_list = keywords.split(', ')[:8]
                        print(f"   块{j}: {', '.join(kw_list)}")

        print("=" * 80)


def main():
    """主函数"""
    print("🚀 初始化混合检索 RAG 系统...")

    config = RAGConfig(
        max_results=10,
        similarity_threshold=0.70,  # 可适当调整
        chunk_size=1000,
        chunk_overlap=250,
        use_hybrid_search=True,
        keyword_boost=0.3  # 30%权重给关键词匹配
    )
    rag = GeminiRAG(
        collection_name="md_documents",
        persist_directory="./chroma_db",
        config=config
    )

    print("\n📖 从 docs 文件夹加载文档...")
    rag.load_documents_from_folder(
        folder_path="./docs",
        force_reload=False,
    )

    rag.list_documents()
    print(f"\n📊 {rag.get_collection_info()}")

    print("\n" + "=" * 70)
    print("🤖 开始测试查询...")
    print("=" * 70)

    queries = [
        "奥克肖特的政治思想",
        "如何评价罗尔斯",
        "格林的政治思想",
        "诺齐克的理论介绍",
    ]

    for query in queries:
        print(f"\n❓ 问题: {query}")
        print("-" * 70)

        search_results = rag.search(query)

        if search_results['documents'][0]:
            print(f"🔍 检索到 {len(search_results['documents'][0])} 个相关文档块:")
            for metadata, distance in zip(search_results['metadatas'][0], search_results['distances'][0]):
                similarity = (1 - distance) * 100
                source = metadata.get('source', 'Unknown')
                chunk_id = metadata.get('chunk_id', '')
                chunk_info = f" [{chunk_id.split('_')[-1]}]" if chunk_id else ""
                print(f"   • {source}{chunk_info} (相似度: {similarity:.1f}%)")

            print(f"\n💡 答案:")
            answer = rag.generate_answer(query)
            print(answer)
        else:
            print("🔍 没有找到满足相似度阈值的相关文档")
            print(f"\n💡 答案:")
            print("❌ 抱歉,在知识库中没有找到相关信息来回答这个问题。")

        print("=" * 70)

    print("\n✨ 演示完成!")
    print("💾 向量数据库已持久化保存,下次运行将自动复用")


# ============ FastAPI 实现 ============

app = FastAPI(title="RAG API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 RAG 实例
rag_instance = None

# 会话管理（存储每个会话的历史）
sessions: Dict[str, list] = {}


def get_rag():
    """获取或初始化 RAG 实例"""
    global rag_instance
    if rag_instance is None:
        config = RAGConfig(
            max_results=10,
            similarity_threshold=0.70,
            chunk_size=1000,
            chunk_overlap=250,
            use_hybrid_search=True,
            keyword_boost=0.3
        )
        rag_instance = GeminiRAG(
            collection_name="md_documents",
            persist_directory="./chroma_db",
            config=config
        )
        rag_instance.load_documents_from_folder(
            folder_path="./docs",
            force_reload=False,
        )
    return rag_instance


# ========== Pydantic Models ==========

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    n_results: Optional[int] = None
    similarity_threshold: Optional[float] = None


class SessionRequest(BaseModel):
    session_id: str


class ReloadRequest(BaseModel):
    force_reload: bool = False


# ========== API 端点 ==========

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "message": "RAG API is running"}


@app.get("/api/info")
async def get_info():
    """获取系统信息"""
    try:
        rag = get_rag()
        count = rag.collection.count()

        results = rag.collection.get()
        docs_by_source = defaultdict(list)
        for metadata in results['metadatas']:
            source = metadata.get('source', 'unknown')
            docs_by_source[source].append(metadata)

        doc_list = []
        for source, metadatas in sorted(docs_by_source.items()):
            doc_list.append({
                'source': source,
                'chunk_count': len(metadatas),
                'file_size': metadatas[0].get('file_size', 0)
            })

        return {
            'success': True,
            'total_chunks': count,
            'total_documents': len(doc_list),
            'documents': doc_list,
            'active_sessions': len(sessions),
            'config': {
                'max_results': rag.config.max_results,
                'similarity_threshold': rag.config.similarity_threshold,
                'use_hybrid_search': rag.config.use_hybrid_search,
                'keyword_boost': rag.config.keyword_boost
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/create")
async def create_session():
    """创建新的会话"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return {
        'success': True,
        'session_id': session_id,
        'message': '会话创建成功'
    }


@app.post("/api/session/clear")
async def clear_session(request: SessionRequest):
    """清空会话历史"""
    session_id = request.session_id
    if session_id in sessions:
        sessions[session_id] = []
        return {
            'success': True,
            'message': '会话历史已清空'
        }
    else:
        raise HTTPException(status_code=404, detail='会话不存在')


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史"""
    if session_id in sessions:
        return {
            'success': True,
            'session_id': session_id,
            'history': sessions[session_id]
        }
    else:
        raise HTTPException(status_code=404, detail='会话不存在')


@app.post("/api/ask/stream")
async def ask_question_stream(request: QueryRequest):
    """RAG 问答接口（流式传输）"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail='问题不能为空')

        session_id = request.session_id

        # 如果没有 session_id，创建新会话
        if not session_id:
            session_id = str(uuid.uuid4())
            sessions[session_id] = []
        elif session_id not in sessions:
            sessions[session_id] = []

        # 获取历史记录
        chat_history = sessions[session_id]

        rag = get_rag()

        async def event_generator():
            # 发送 session_id
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"

            # 流式生成答案
            full_answer = ""
            for chunk_data in rag.generate_answer_stream(
                    query,
                    chat_history=chat_history,
                    n_results=request.n_results,
                    similarity_threshold=request.similarity_threshold
            ):
                yield f"data: {chunk_data}\n\n"

                # 收集完整答案
                chunk_obj = json.loads(chunk_data)
                if chunk_obj['type'] == 'content':
                    full_answer += chunk_obj['content']

                await asyncio.sleep(0.01)  # 小延迟，让流更平滑

            # 更新会话历史
            sessions[session_id].append({
                'role': 'user',
                'parts': [{'text': query}]
            })
            sessions[session_id].append({
                'role': 'model',
                'parts': [{'text': full_answer}]
            })

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


@app.post("/api/search")
async def search_documents(request: QueryRequest):
    """搜索相关文档（非流式）"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail='查询内容不能为空')

        rag = get_rag()
        search_results = rag.search(
            query,
            request.n_results,
            request.similarity_threshold
        )

        results = []
        if search_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                    search_results['documents'][0],
                    search_results['metadatas'][0],
                    search_results['distances'][0]
            )):
                keyword_score = search_results.get('keyword_scores', [[]])[0][
                    i] if 'keyword_scores' in search_results else 0

                results.append({
                    'document': doc,
                    'source': metadata.get('source', ''),
                    'section_path': metadata.get('section_path', ''),
                    'keywords': metadata.get('keywords', ''),
                    'chunk_id': metadata.get('chunk_id', ''),
                    'semantic_similarity': round((1 - distance) * 100, 2),
                    'keyword_score': round(keyword_score * 100, 2)
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
        rag = get_rag()
        rag.load_documents_from_folder(
            folder_path="./docs",
            force_reload=request.force_reload
        )

        return {
            'success': True,
            'message': '文档重新加载完成',
            'total_chunks': rag.collection.count()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_fastapi(host='0.0.0.0', port=8000):
    """启动 FastAPI 服务"""
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 启动 FastAPI 服务...")
    print("=" * 70)

    print("\n📦 预加载 RAG 系统...")
    get_rag()

    print("\n✅ RAG 系统初始化完成")
    print(f"🌐 API 服务运行在: http://{host}:{port}")
    print(f"📚 API 文档: http://{host}:{port}/docs")
    print("\n📡 可用的 API 端点:")
    print(f"   • GET  /api/health              - 健康检查")
    print(f"   • GET  /api/info                - 系统信息")
    print(f"   • POST /api/session/create      - 创建会话")
    print(f"   • POST /api/session/clear       - 清空会话")
    print(f"   • GET  /api/session/:id/history - 获取历史")
    print(f"   • POST /api/ask/stream          - RAG 问答（流式）")
    print(f"   • POST /api/search              - 搜索文档")
    print(f"   • POST /api/reload              - 重新加载文档")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    start_fastapi(host='0.0.0.0', port=8000)
    #if len(sys.argv) > 1 and sys.argv[1] == 'api':
        #start_fastapi(host='0.0.0.0', port=8000)
    #else:
        # 原来的测试代码
        #main()