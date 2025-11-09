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

from starlette.staticfiles import StaticFiles

from implements.RAGConfig import RAGConfig

API_KEY = 'AIzaSyCNwmo17IETTpEAhCp9mvrtaovXteITZDM'

genai.configure(api_key=API_KEY)
os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"


# 请将下面的整个 class GeminiRAG 替换掉您文件中的同名 class

class GeminiRAG:
    """
    一个集成了查询重写、增强嵌入、上下文窗口扩展和多轮对话历史的
    高级检索增强生成 (RAG) 系统。
    """

    def __init__(
            self,
            collection_name: str = "documents_optimized",
            persist_directory: str = "./chroma_db_optimized",
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

        print("📦 加载嵌入模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ 嵌入模型加载完成")

        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')

        print(f"\n⚙️  RAG 系统配置:")
        print(f"   ├─ 检索策略: {'混合检索 (关键词+语义)' if self.config.use_hybrid_search else '纯语义检索'}")
        if self.config.use_hybrid_search:
            print(f"   │  └─ 关键词权重: {self.config.keyword_boost}")
        print(f"   ├─ 查询重写优化: {'✅ 已启用' if self.config.use_query_rewriting else '❌ 未启用'}")
        print(
            f"   ├─ 上下文窗口扩展: {'✅ 已启用 (窗口大小: ' + str(self.config.context_window_size) + ')' if self.config.context_window_size > 1 else '❌ 未启用'}")
        print(f"   ├─ 最大检索数: {self.config.max_results}")
        print(f"   ├─ 相似度阈值: {self.config.similarity_threshold or '自动'}")
        print(f"   ├─ 分块大小: {self.config.chunk_size} 字符")
        print(f"   └─ 重叠大小: {self.config.chunk_overlap} 字符")

    # ... [从 _get_embedding 到 _smart_chunk_document 的所有辅助函数保持不变] ...
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
        hashtags = re.findall(r'#([^\s#]+)', text)
        keywords.extend(hashtags)
        bold_text = re.findall(r'\*\*([^*]+)\*\*', text)
        keywords.extend([t.strip() for t in bold_text if 3 < len(t.strip()) < 30])
        headers = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        keywords.extend([h.strip() for h in headers if len(h.strip()) < 50])
        names = re.findall(r'([·\u4e00-\u9fa5]{2,6}(?:的|、|与|和|及)?)', text)
        potential_names = [n.strip('的、与和及') for n in names if 2 <= len(n.strip('的、与和及')) <= 6]
        keywords.extend(potential_names)
        return list(set(keywords))

    def _extract_query_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词，用于混合检索评分"""
        keywords = set(re.findall(r'[\u4e00-\u9fa5·]{2,6}', query))
        key_phrases = ['政治思想', '理论', '观点', '学说', '主张', '批判', '评价', '正义论']
        for phrase in key_phrases:
            if phrase in query:
                keywords.add(phrase)
        return list(keywords)

    def _keyword_match_score(self, query: str, doc_text: str, doc_keywords: str) -> float:
        """计算关键词匹配分数"""
        score = 0.0
        query_keywords = self._extract_query_keywords(query)
        if not query_keywords:
            return 0.0
        doc_text_lower = doc_text.lower()
        doc_keywords_lower = doc_keywords.lower() if doc_keywords else ""
        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in doc_keywords_lower:
                score += 0.5
            count = min(doc_text_lower.count(keyword_lower), 5)
            score += count * 0.1
        return min(score, 1.0)

    def _parse_document_structure(self, text: str) -> List[Dict]:
        """解析文档结构，识别标题和章节"""
        lines = text.split('\n')
        sections = []
        current_section = {'content': [], 'headers': [], 'line_start': 0, 'header_level': 0}
        header_stack = []
        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section)
                level, title = len(header_match.group(1)), header_match.group(2).strip()
                while header_stack and header_stack[-1]['level'] >= level:
                    header_stack.pop()
                header_stack.append({'level': level, 'title': title})
                current_section = {
                    'content': [line], 'headers': [h['title'] for h in header_stack],
                    'header_level': level, 'line_start': i
                }
            else:
                current_section['content'].append(line)
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)
        return sections

    def _split_long_text(self, text: str, max_size: int) -> List[str]:
        """当单个章节内容过长时，按句子进行分割"""
        if len(text) <= max_size:
            return [text]
        sentences = re.split(r'([。！？\n]+)', text)
        parts = []
        current_part = ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
            if len(current_part) + len(sentence) + len(delimiter) > max_size and current_part:
                parts.append(current_part)
                overlap_start = max(0, len(current_part) - self.config.chunk_overlap)
                current_part = current_part[overlap_start:] + sentence + delimiter
            else:
                current_part += sentence + delimiter
        if current_part:
            parts.append(current_part)
        return parts if parts else [text[:max_size]]

    def _smart_chunk_document(self, text: str, file_name: str) -> List[tuple]:
        """智能分块文档：基于Markdown结构，合并小章节，分割大章节"""
        sections = self._parse_document_structure(text)
        chunks = []
        chunk_idx = 0
        i = 0
        while i < len(sections):
            headers_text = ' > '.join(sections[i].get('headers', []))
            chunk_content_parts = []
            current_size = 0
            start_section_idx = i
            j = i
            while j < len(sections):
                section_to_add = sections[j]
                content_to_add = section_to_add['content']
                if j > start_section_idx and section_to_add.get('header_level', 6) <= 2:
                    break
                if current_size > 0 and current_size + len(content_to_add) > self.config.chunk_size:
                    break
                chunk_content_parts.append(content_to_add)
                current_size += len(content_to_add)
                j += 1
            full_content = "\n\n".join(chunk_content_parts)
            split_parts = self._split_long_text(full_content, self.config.chunk_size) if len(
                full_content) > self.config.chunk_size else [full_content]
            for part in split_parts:
                keywords = self._extract_keywords(f"# {headers_text}\n{part}")
                chunk_meta = {'section_path': headers_text, 'keywords': ', '.join(keywords)}
                chunk_id = f"{file_name}_chunk_{chunk_idx}"
                chunks.append((part, chunk_id, len(part), chunk_meta))
                chunk_idx += 1
            i = j - 1 if j > start_section_idx + 1 else j
        return chunks

    def load_documents_from_folder(self, folder_path: str = "./docs", force_reload: bool = False):
        """从文件夹递归加载所有 Markdown 文档 (包含嵌入优化和路径修复)"""
        # [FIX] 使用 resolve() 获取绝对路径，确保路径一致性
        docs_path = Path(folder_path).resolve()
        if not docs_path.exists():
            print(f"❌ 文件夹不存在: {docs_path}")
            return

        md_files = list(docs_path.rglob("*.md"))
        if not md_files:
            print(f"⚠️  在 {docs_path} 中没有找到 Markdown 文件。")
            return

        print(f"\n📁 找到 {len(md_files)} 个 Markdown 文件，开始处理...")

        total_docs_in_db = self.collection.count()
        all_metadatas = self.collection.get(limit=total_docs_in_db, include=["metadatas"])[
            'metadatas'] if total_docs_in_db > 0 else []
        existing_hashes = {meta.get('source'): meta.get('file_hash') for meta in all_metadatas if
                           'source' in meta and 'file_hash' in meta}

        new_docs_content, new_docs_for_embedding, new_ids, new_metadatas = [], [], [], []
        ids_to_delete = []
        updated_count, new_count, skipped_count = 0, 0, 0

        for md_file in md_files:
            # [FIX] 确保 relative_path 使用绝对路径作为基准
            relative_path = str(md_file.relative_to(docs_path)).replace('\\', '/')
            current_hash = self._get_file_hash(str(md_file))

            if relative_path in existing_hashes:
                if not force_reload and existing_hashes[relative_path] == current_hash:
                    skipped_count += 1
                    continue
                else:
                    print(f"🔄 检测到文件变更: {relative_path}")
                    updated_count += 1
                    results = self.collection.get(where={"source": relative_path}, include=[])
                    ids_to_delete.extend(results['ids'])
            else:
                new_count += 1

            content = self._read_markdown_file(md_file)
            if content:
                chunks = self._smart_chunk_document(content, relative_path)
                for chunk_text, chunk_id, chunk_size, chunk_meta in chunks:
                    new_docs_content.append(chunk_text)
                    new_ids.append(chunk_id)
                    embedding_text = f"所属章节: {chunk_meta.get('section_path', '')}\n关键词: {chunk_meta.get('keywords', '')}\n\n内容:\n{chunk_text}"
                    new_docs_for_embedding.append(embedding_text)
                    metadata = {"source": relative_path, "file_hash": current_hash, **chunk_meta}
                    new_metadatas.append(metadata)

        if ids_to_delete:
            print(f"🗑️  正在删除 {len(ids_to_delete)} 个旧的文档块...")
            delete_batch_size = 500
            for i in range(0, len(ids_to_delete), delete_batch_size):
                self.collection.delete(ids=ids_to_delete[i:i + delete_batch_size])
            print("✅ 旧文档块删除完毕。")

        if new_docs_content:
            print(
                f"\n💾 准备处理 {new_count} 个新文件和 {updated_count} 个更新文件，共计 {len(new_docs_content)} 个新文档块...")
            print("🔄 生成增强嵌入向量...")
            embedding_batch_size = 32
            embeddings = []
            for i in range(0, len(new_docs_for_embedding), embedding_batch_size):
                batch = new_docs_for_embedding[i:i + embedding_batch_size]
                batch_embeddings = self.embed_model.encode(batch, convert_to_tensor=False,
                                                           show_progress_bar=False).tolist()
                embeddings.extend(batch_embeddings)
                print(
                    f"   生成嵌入向量进度: {min(i + embedding_batch_size, len(new_docs_for_embedding))}/{len(new_docs_for_embedding)}")

            db_batch_size = 4000
            total_batches = (len(new_ids) + db_batch_size - 1) // db_batch_size
            print(f"\n➕ 正在将 {len(new_ids)} 个文档块分 {total_batches} 批添加到数据库...")
            for i in range(0, len(new_ids), db_batch_size):
                self.collection.add(
                    ids=new_ids[i:i + db_batch_size],
                    documents=new_docs_content[i:i + db_batch_size],
                    embeddings=embeddings[i:i + db_batch_size],
                    metadatas=new_metadatas[i:i + db_batch_size]
                )
                print(f"   批次 {i // db_batch_size + 1}/{total_batches} 添加成功。")
            print(f"✅ 成功添加/更新 {len(new_docs_content)} 个文档块。")

        if skipped_count > 0:
            print(f"⏭️  跳过 {skipped_count} 个未修改的文档。")
        print(f"\n📊 数据库当前共有 {self.collection.count()} 个文档块。")

    def _rewrite_query_for_retrieval(self, query: str) -> str:
        if not self.config.use_query_rewriting:
            return query
        print(f"\n🔄 正在重写查询...")
        prompt = f"""你是一名检索优化专家。请将以下用户问题改写为一个信息更丰富的陈述句，用于向量数据库的语义检索。请专注于核心意图，补充可能的上下文，使其更像一个“答案”的片段。直接返回改写后的文本，不要包含任何解释或前缀。原始问题: "{query}"\n\n改写后的检索查询:"""
        try:
            response = self.chat_model.generate_content(prompt)
            rewritten_query = response.text.strip().replace("*", "")
            print(f"   - 原始查询: {query}")
            print(f"   - 重写后: {rewritten_query}")
            return rewritten_query
        except Exception as e:
            print(f"⚠️ 查询重写失败: {e}，将使用原始查询。")
            return query

    def search(self, query: str) -> dict:
        rewritten_query = self._rewrite_query_for_retrieval(query)
        query_embedding = self._get_embedding(rewritten_query)
        search_n = min(self.config.max_results * 5, self.collection.count())
        if search_n == 0:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_n,
            include=["metadatas", "documents", "distances"]
        )
        if not semantic_results['documents'][0]:
            return semantic_results

        if self.config.use_hybrid_search:
            scored_results = []
            for doc_id, doc, meta, dist in zip(semantic_results['ids'][0], semantic_results['documents'][0],
                                               semantic_results['metadatas'][0], semantic_results['distances'][0]):
                semantic_score = 1 - dist
                keyword_score = self._keyword_match_score(query, doc, meta.get('keywords', ''))
                final_score = semantic_score * (
                            1 - self.config.keyword_boost) + keyword_score * self.config.keyword_boost
                scored_results.append({'id': doc_id, 'doc': doc, 'meta': meta, 'dist': dist, 'score': final_score})
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            filtered = [r for r in scored_results if (1 - r['dist']) >= self.config.similarity_threshold]
            top_results = filtered[:self.config.max_results]
            return {
                'ids': [[r['id'] for r in top_results]], 'documents': [[r['doc'] for r in top_results]],
                'metadatas': [[r['meta'] for r in top_results]], 'distances': [[r['dist'] for r in top_results]]
            }

        indices = [i for i, d in enumerate(semantic_results['distances'][0]) if
                   (1 - d) >= self.config.similarity_threshold]
        top_indices = indices[:self.config.max_results]
        return {
            'ids': [[semantic_results['ids'][0][i] for i in top_indices]],
            'documents': [[semantic_results['documents'][0][i] for i in top_indices]],
            'metadatas': [[semantic_results['metadatas'][0][i] for i in top_indices]],
            'distances': [[semantic_results['distances'][0][i] for i in top_indices]]
        }

    def _expand_context_with_window(self, search_results: dict) -> List[Dict]:
        if self.config.context_window_size <= 1 or not search_results['ids'][0]:
            return [{"doc": doc, "meta": meta, "is_hit": True} for doc, meta in
                    zip(search_results['documents'][0], search_results['metadatas'][0])]

        print("🔄 正在扩展上下文窗口...")
        final_docs = {r_id: {"doc": doc, "meta": meta, "is_hit": True} for r_id, doc, meta in
                      zip(search_results['ids'][0], search_results['documents'][0], search_results['metadatas'][0])}
        ids_to_fetch = set()
        window_radius = self.config.context_window_size // 2
        for r_id in search_results['ids'][0]:
            parts = r_id.rsplit('_chunk_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name, index = parts[0], int(parts[1])
                for i in range(1, window_radius + 1):
                    prev_id = f"{base_name}_chunk_{index - i}"
                    next_id = f"{base_name}_chunk_{index + i}"
                    if prev_id not in final_docs: ids_to_fetch.add(prev_id)
                    if next_id not in final_docs: ids_to_fetch.add(next_id)
        if ids_to_fetch:
            ids_list = list(ids_to_fetch)
            get_batch_size = 500
            for i in range(0, len(ids_list), get_batch_size):
                batch_ids = ids_list[i:i + get_batch_size]
                context_docs = self.collection.get(ids=batch_ids, include=["metadatas", "documents"])
                for c_id, doc, meta in zip(context_docs['ids'], context_docs['documents'], context_docs['metadatas']):
                    if c_id not in final_docs:
                        final_docs[c_id] = {"doc": doc, "meta": meta, "is_hit": False}
        sorted_ids = sorted(final_docs.keys(),
                            key=lambda x: (final_docs[x]['meta']['source'], int(x.rsplit('_', 1)[1])))
        return [final_docs[id] for id in sorted_ids]

    # ============== [核心修改区域] ==============
    # 用下面这个实现了多轮对话的版本，替换您当前的 generate_answer_stream 和 generate_answer 方法
    # ============================================

    def generate_answer_stream(self, query: str, chat_history: list = None):
        """
        RAG 流式生成答案（支持多轮对话）。
        此方法会先检索与当前问题相关的文档，然后将这些文档作为上下文，
        连同历史对话记录一起，交给大模型生成回答。
        """
        # 步骤 1: 检索与当前问题相关的文档
        search_results = self.search(query)
        has_results = (search_results and search_results.get('documents') and search_results['documents'][0])

        if not has_results:
            yield json.dumps({'type': 'error', 'content': '没有找到相关文档,无法回答问题。'}, ensure_ascii=False) + '\n'
            return

        # 步骤 2: 扩展上下文窗口，获取更完整的文档片段
        context_items = self._expand_context_with_window(search_results)

        # 步骤 3: 准备并发送 "sources" 信息
        sources_info = [{'source': item['meta']['source'], 'section_path': item['meta'].get('section_path', ''),
                         'is_hit': item['is_hit']} for item in context_items]
        yield json.dumps({'type': 'sources', 'content': sources_info, 'count': len(sources_info)},
                         ensure_ascii=False) + '\n'

        # 步骤 4: 构建仅包含当前检索到的文档的上下文文本
        context_parts = []
        for item in context_items:
            source_info = f"[来源: {item['meta'].get('source', '未知')} | 章节: {item['meta'].get('section_path', 'N/A')}]"
            context_parts.append(f"{source_info}\n{item['doc']}")
        context_text = "\n\n---\n\n".join(context_parts)
        print(f"\n📚 本轮检索到 {len(context_parts)} 个文档块作为上下文。")

        # 步骤 5: 创建一个有状态的对话实例，并载入历史记录
        chat = self.chat_model.start_chat(history=chat_history or [])

        # 步骤 6: 构建发送给模型的最终消息
        # 这条消息包含了系统指令、本轮检索到的上下文和当前用户的问题
        user_message = f"""你是一个专业的政治学知识解答模型。请严格基于以下最新检索到的文档内容，并结合之前的对话历史，以系统、学术化的方式回答用户当前的问题。以Markdown格式进行回复。

--- [最新检索到的文档 (用于回答当前问题)] ---
{context_text}
--- [检索到的文档结束] ---

当前问题: {query}
"""

        # 步骤 7: 流式发送消息并返回结果
        try:
            response = chat.send_message(user_message, stream=True)
            for chunk in response:
                if chunk.text:
                    yield json.dumps({'type': 'content', 'content': chunk.text}, ensure_ascii=False) + '\n'
            yield json.dumps({'type': 'done', 'content': ''}, ensure_ascii=False) + '\n'
        except Exception as e:
            yield json.dumps({'type': 'error', 'content': f'生成答案时出错: {str(e)}'}, ensure_ascii=False) + '\n'

    def generate_answer(self, query: str) -> str:
        """RAG非流式生成答案（为保持兼容性而保留，但不支持多轮对话）"""
        full_response = ""
        # 非流式方法本质上是流式方法的聚合
        for chunk_data in self.generate_answer_stream(query, chat_history=None):
            chunk = json.loads(chunk_data)
            if chunk['type'] == 'content':
                full_response += chunk['content']
            elif chunk['type'] == 'error':
                return f"❌ {chunk['content']}"
        return full_response

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


app.mount("/", StaticFiles(directory="web", html=True), name="web")

# 也可以用以下方式，显式地为根路径提供index.html
# @app.get("/")
# async def read_index():
#    return FileResponse('web/index.html')


def start_fastapi(host='0.0.0.0', port=8000):
    """启动 FastAPI 服务"""
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 启动 FastAPI 服务...")
    print("=" * 70)

    print("\n📦 预加载 RAG 系统...")
    get_rag()

    print("\n✅ RAG 系统初始化完成")
    print(f"🌐 Web UI 访问地址: http://{host}:{port}")
    print(f"📚 API 文档 (Swagger): http://{host}:{port}/docs")
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
    # 移除原有的main()函数调用和命令行参数判断
    # 直接启动FastAPI服务
    start_fastapi(host='0.0.0.0', port=8000)