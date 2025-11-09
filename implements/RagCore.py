import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


class RAGConfig:
    """RAG 系统配置类"""

    def __init__(
            self,
            # 检索相关配置
            max_results: int = 7,
            similarity_threshold: float = 0.5,
            use_hybrid_search: bool = True,
            keyword_boost: float = 0.35,

            # 文档处理配置
            chunk_size: int = 1000,
            chunk_overlap: int = 200,

            # 上下文优化配置
            context_window_size: int = 3,
    ):
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_search = use_hybrid_search
        self.keyword_boost = keyword_boost
        self.context_window_size = context_window_size


class RAGRetriever:
    """
    RAG 检索核心模块 - 只负责文档存储、索引和检索
    不涉及任何AI对话功能
    """

    def __init__(
            self,
            collection_name: str = "documents_optimized",
            persist_directory: str = "./chroma_db_optimized",
            config: Optional[RAGConfig] = None
    ):
        """初始化 RAG 检索器"""
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

        print(f"\n⚙️  RAG 检索器配置:")
        print(f"   ├─ 检索策略: {'混合检索 (关键词+语义)' if self.config.use_hybrid_search else '纯语义检索'}")
        if self.config.use_hybrid_search:
            print(f"   │  └─ 关键词权重: {self.config.keyword_boost}")
        print(
            f"   ├─ 上下文窗口扩展: {'✅ 已启用 (窗口大小: ' + str(self.config.context_window_size) + ')' if self.config.context_window_size > 1 else '❌ 未启用'}")
        print(f"   ├─ 最大检索数: {self.config.max_results}")
        print(f"   ├─ 相似度阈值: {self.config.similarity_threshold or '自动'}")
        print(f"   ├─ 分块大小: {self.config.chunk_size} 字符")
        print(f"   └─ 重叠大小: {self.config.chunk_overlap} 字符")

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
        """从查询中提取关键词"""
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
                    'content': [line],
                    'headers': [h['title'] for h in header_stack],
                    'header_level': level,
                    'line_start': i
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
        """智能分块文档：基于Markdown结构"""
        sections = self._parse_document_structure(text)
        chunks = []
        chunk_idx = 0

        i = 0
        while i < len(sections):
            current_section = sections[i]
            headers_text = ' > '.join(current_section.get('headers', []))

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

            if len(full_content) > self.config.chunk_size:
                split_parts = self._split_long_text(full_content, self.config.chunk_size)
            else:
                split_parts = [full_content]

            for part in split_parts:
                keywords = self._extract_keywords(f"# {headers_text}\n{part}")
                chunk_meta = {
                    'section_path': headers_text,
                    'keywords': ', '.join(keywords),
                }
                chunk_id = f"{file_name}_chunk_{chunk_idx}"
                chunks.append((part, chunk_id, len(part), chunk_meta))
                chunk_idx += 1

            if j > start_section_idx + 1:
                i = j - 1
            else:
                i = j

        return chunks

    def load_documents_from_folder(self, folder_path: str = "./docs", force_reload: bool = False):
        """从文件夹递归加载所有 Markdown 文档"""
        docs_path = Path(folder_path)
        if not docs_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return

        md_files = list(docs_path.rglob("*.md"))
        if not md_files:
            print(f"⚠️  在 {folder_path} 中没有找到 Markdown 文件。")
            return

        print(f"\n📁 找到 {len(md_files)} 个 Markdown 文件，开始处理...")

        total_docs_in_db = self.collection.count()
        all_metadatas = self.collection.get(limit=total_docs_in_db, include=["metadatas"])[
            'metadatas'] if total_docs_in_db > 0 else []
        existing_hashes = {meta.get('source'): meta.get('file_hash') for meta in all_metadatas if
                           'source' in meta and 'file_hash' in meta}

        new_docs_content = []
        new_docs_for_embedding = []
        new_ids = []
        new_metadatas = []
        ids_to_delete = []
        updated_count, new_count, skipped_count = 0, 0, 0

        for md_file in md_files:
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

                    metadata = {
                        "source": relative_path,
                        "file_hash": current_hash,
                        **chunk_meta
                    }
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
            print("🔄 生成增强嵌入向量（可能需要一些时间）...")

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
                batch_ids = new_ids[i:i + db_batch_size]
                batch_documents = new_docs_content[i:i + db_batch_size]
                batch_embeddings = embeddings[i:i + db_batch_size]
                batch_metadatas = new_metadatas[i:i + db_batch_size]

                self.collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                print(f"   批次 {i // db_batch_size + 1}/{total_batches} 添加成功。")

            print(f"✅ 成功添加/更新 {len(new_docs_content)} 个文档块。")

        if skipped_count > 0:
            print(f"⏭️  跳过 {skipped_count} 个未修改的文档。")

        print(f"\n📊 数据库当前共有 {self.collection.count()} 个文档块。")

    def search(self, query: str, n_results: Optional[int] = None) -> dict:
        """混合检索"""
        query_embedding = self._get_embedding(query)
        n_results = n_results or self.config.max_results
        search_n = min(n_results * 5, self.collection.count())

        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_n,
            include=["metadatas", "documents", "distances"]
        )

        if not semantic_results['documents'][0]:
            return semantic_results

        if self.config.use_hybrid_search:
            scored_results = []
            for doc_id, doc, meta, dist in zip(
                    semantic_results['ids'][0],
                    semantic_results['documents'][0],
                    semantic_results['metadatas'][0],
                    semantic_results['distances'][0]
            ):
                semantic_score = 1 - dist
                keyword_score = self._keyword_match_score(query, doc, meta.get('keywords', ''))
                final_score = semantic_score * (
                            1 - self.config.keyword_boost) + keyword_score * self.config.keyword_boost
                scored_results.append({
                    'id': doc_id,
                    'doc': doc,
                    'meta': meta,
                    'dist': dist,
                    'score': final_score
                })

            scored_results.sort(key=lambda x: x['score'], reverse=True)
            filtered = [r for r in scored_results if (1 - r['dist']) >= self.config.similarity_threshold]
            top_results = filtered[:n_results]

            return {
                'ids': [[r['id'] for r in top_results]],
                'documents': [[r['doc'] for r in top_results]],
                'metadatas': [[r['meta'] for r in top_results]],
                'distances': [[r['dist'] for r in top_results]],
            }

        indices = [i for i, d in enumerate(semantic_results['distances'][0]) if
                   (1 - d) >= self.config.similarity_threshold]
        top_indices = indices[:n_results]
        return {
            'ids': [[semantic_results['ids'][0][i] for i in top_indices]],
            'documents': [[semantic_results['documents'][0][i] for i in top_indices]],
            'metadatas': [[semantic_results['metadatas'][0][i] for i in top_indices]],
            'distances': [[semantic_results['distances'][0][i] for i in top_indices]],
        }

    def expand_context_with_window(self, search_results: dict) -> List[Dict]:
        """扩展检索结果的上下文窗口"""
        if self.config.context_window_size <= 1 or not search_results['ids'][0]:
            return [{"doc": doc, "meta": meta, "is_hit": True} for doc, meta in
                    zip(search_results['documents'][0], search_results['metadatas'][0])]

        print("🔄 正在扩展上下文窗口...")
        final_docs = {
            r_id: {"doc": doc, "meta": meta, "is_hit": True}
            for r_id, doc, meta in zip(
                search_results['ids'][0],
                search_results['documents'][0],
                search_results['metadatas'][0]
            )
        }

        ids_to_fetch = set()
        window_radius = self.config.context_window_size // 2

        for r_id in search_results['ids'][0]:
            parts = r_id.rsplit('_chunk_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name, index = parts[0], int(parts[1])
                for i in range(1, window_radius + 1):
                    prev_id = f"{base_name}_chunk_{index - i}"
                    next_id = f"{base_name}_chunk_{index + i}"
                    if prev_id not in final_docs:
                        ids_to_fetch.add(prev_id)
                    if next_id not in final_docs:
                        ids_to_fetch.add(next_id)

        if ids_to_fetch:
            ids_list = list(ids_to_fetch)
            get_batch_size = 500
            for i in range(0, len(ids_list), get_batch_size):
                batch_ids = ids_list[i:i + get_batch_size]
                context_docs = self.collection.get(ids=batch_ids, include=["metadatas", "documents"])
                for c_id, doc, meta in zip(context_docs['ids'], context_docs['documents'], context_docs['metadatas']):
                    if c_id not in final_docs:
                        final_docs[c_id] = {"doc": doc, "meta": meta, "is_hit": False}

        sorted_ids = sorted(
            final_docs.keys(),
            key=lambda x: (final_docs[x]['meta']['source'], int(x.rsplit('_', 1)[1]))
        )
        return [final_docs[id] for id in sorted_ids]

    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        count = self.collection.count()
        results = self.collection.get()
        docs_by_source = defaultdict(list)

        for metadata in results['metadatas']:
            source = metadata.get('source', 'unknown')
            docs_by_source[source].append(metadata)

        doc_list = []
        for source, metadatas in sorted(docs_by_source.items()):
            doc_list.append({
                'source': source,
                'chunk_count': len(metadatas),
            })

        return {
            'total_chunks': count,
            'total_documents': len(doc_list),
            'documents': doc_list
        }