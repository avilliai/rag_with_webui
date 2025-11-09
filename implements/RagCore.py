import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# PDF和EPUB支持
try:
    import PyPDF2
    import fitz  # PyMuPDF

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  未安装PDF库,请运行: pip install PyPDF2 PyMuPDF")

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    EPUB_SUPPORT = True
except ImportError:
    EPUB_SUPPORT = False
    print("⚠️  未安装EPUB库,请运行: pip install ebooklib beautifulsoup4")


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

    def _clean_text(self, text: str) -> str:
        """清理文本中的异常字符和格式问题"""
        if not text:
            return ""

        # 移除空字符和控制字符（保留换行符、制表符）
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

        # 规范化空白字符
        text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符变为一个空格
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # 多个空行变为两个换行

        # 移除页眉页脚常见模式
        text = re.sub(r'第\s*\d+\s*页.*?共\s*\d+\s*页', '', text)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)

        return text.strip()

    def _read_pdf_file(self, file_path: Path) -> str:
        """读取PDF文件,使用多种方法确保健壮性"""
        if not PDF_SUPPORT:
            print(f"⚠️  跳过PDF文件 {file_path}: 未安装PDF支持库")
            return None

        text = ""

        # 方法1: 使用PyMuPDF (fitz) - 对中文支持更好
        try:
            doc = fitz.open(str(file_path))
            for page_num, page in enumerate(doc, 1):
                try:
                    page_text = page.get_text()
                    if page_text.strip():
                        text += f"\n\n--- 第 {page_num} 页 ---\n\n{page_text}"
                except Exception as e:
                    print(f"   ⚠️  页面 {page_num} 提取失败: {e}")
                    continue
            doc.close()

            if text.strip():
                return self._clean_text(text)
        except Exception as e:
            print(f"   ⚠️  PyMuPDF提取失败: {e}, 尝试备用方法...")

        # 方法2: 使用PyPDF2作为备用
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n\n--- 第 {page_num + 1} 页 ---\n\n{page_text}"
                    except Exception as e:
                        print(f"   ⚠️  页面 {page_num + 1} 提取失败: {e}")
                        continue

            if text.strip():
                return self._clean_text(text)
        except Exception as e:
            print(f"   ⚠️  PyPDF2提取失败: {e}")

        # 如果两种方法都失败
        if not text.strip():
            print(f"   ❌ 无法从PDF提取文本: {file_path}")
            return None

        return self._clean_text(text)

    def _read_epub_file(self, file_path: Path) -> str:
        """读取EPUB文件"""
        if not EPUB_SUPPORT:
            print(f"⚠️  跳过EPUB文件 {file_path}: 未安装EPUB支持库")
            return None

        try:
            book = epub.read_epub(str(file_path))
            text_parts = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    try:
                        content = item.get_content().decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(content, 'html.parser')

                        # 移除script和style标签
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # 提取文本
                        text = soup.get_text(separator='\n', strip=True)
                        if text.strip():
                            text_parts.append(text)
                    except Exception as e:
                        print(f"   ⚠️  EPUB章节解析失败: {e}")
                        continue

            if not text_parts:
                print(f"   ❌ EPUB文件为空: {file_path}")
                return None

            full_text = "\n\n".join(text_parts)
            return self._clean_text(full_text)

        except Exception as e:
            print(f"   ❌ 读取EPUB失败 {file_path}: {e}")
            return None

    def _read_markdown_file(self, file_path: Path) -> str:
        """读取 Markdown 文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  读取文件失败 {file_path}: {e}")
            return None

    def _read_file(self, file_path: Path) -> str:
        """根据文件类型读取文件内容"""
        suffix = file_path.suffix.lower()

        if suffix == '.md':
            return self._read_markdown_file(file_path)
        elif suffix == '.pdf':
            return self._read_pdf_file(file_path)
        elif suffix == '.epub':
            return self._read_epub_file(file_path)
        else:
            print(f"⚠️  不支持的文件类型: {file_path}")
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
        """从文件夹递归加载所有支持的文档 - 智能增量更新版本"""
        docs_path = Path(folder_path)
        if not docs_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return

        # 支持的文件扩展名
        supported_extensions = ['.md', '.pdf', '.epub']
        all_files = []
        for ext in supported_extensions:
            all_files.extend(docs_path.rglob(f"*{ext}"))

        if not all_files:
            print(f"⚠️  在 {folder_path} 中没有找到支持的文档文件。")
            return

        print(f"\n📁 找到 {len(all_files)} 个文档文件，开始智能增量分析...")
        print(f"   支持格式: {', '.join(supported_extensions)}")

        # ========== 第一步: 获取数据库现有文件状态 ==========
        total_docs_in_db = self.collection.count()
        all_metadatas = self.collection.get(limit=total_docs_in_db, include=["metadatas"])[
            'metadatas'] if total_docs_in_db > 0 else []

        # 构建数据库中的文件映射: {相对路径: 哈希值}
        existing_files_in_db = {}
        for meta in all_metadatas:
            if 'source' in meta and 'file_hash' in meta:
                existing_files_in_db[meta['source']] = meta['file_hash']

        print(f"   数据库中现有 {len(existing_files_in_db)} 个文档记录")

        # ========== 第二步: 扫描文件系统,构建当前文件状态 ==========
        current_files = {}  # {相对路径: (完整路径, 哈希值)}
        for doc_file in all_files:
            relative_path = str(doc_file.relative_to(docs_path)).replace('\\', '/')
            current_hash = self._get_file_hash(str(doc_file))
            current_files[relative_path] = (doc_file, current_hash)

        # ========== 第三步: 分类文件状态 ==========
        files_to_add = []  # 新增文件
        files_to_update = []  # 修改文件
        files_to_delete = []  # 删除文件
        files_unchanged = []  # 未变化文件

        # 检测新增和修改
        for relative_path, (doc_file, current_hash) in current_files.items():
            if relative_path not in existing_files_in_db:
                files_to_add.append((relative_path, doc_file, current_hash))
            elif existing_files_in_db[relative_path] != current_hash:
                files_to_update.append((relative_path, doc_file, current_hash))
            else:
                files_unchanged.append(relative_path)

        # 检测删除 (数据库中有,但文件系统中没有)
        for db_path in existing_files_in_db.keys():
            if db_path not in current_files:
                files_to_delete.append(db_path)

        # ========== 第四步: 输出变更摘要 ==========
        print(f"\n📊 文件变更分析:")
        print(f"   ✅ 未变化: {len(files_unchanged)} 个")
        print(f"   ➕ 新增:   {len(files_to_add)} 个")
        print(f"   🔄 修改:   {len(files_to_update)} 个")
        print(f"   🗑️  删除:   {len(files_to_delete)} 个")

        # 如果没有任何变更且不强制重载,直接返回
        if not force_reload and not files_to_add and not files_to_update and not files_to_delete:
            print(f"\n✅ 所有文档都是最新的，无需处理。")
            print(f"📊 数据库当前共有 {self.collection.count()} 个文档块。")
            return

        # ========== 第五步: 处理删除的文件 ==========
        if files_to_delete:
            print(f"\n🗑️  正在删除 {len(files_to_delete)} 个已删除文档...")
            ids_to_delete = []
            for del_path in files_to_delete:
                print(f"   🗑️  删除: {del_path}")
                results = self.collection.get(where={"source": del_path}, include=[])
                ids_to_delete.extend(results['ids'])

            if ids_to_delete:
                delete_batch_size = 500
                for i in range(0, len(ids_to_delete), delete_batch_size):
                    self.collection.delete(ids=ids_to_delete[i:i + delete_batch_size])
                print(f"✅ 已删除 {len(ids_to_delete)} 个文档块")

        # ========== 第六步: 处理更新的文件 ==========
        if files_to_update:
            print(f"\n🔄 正在处理 {len(files_to_update)} 个修改的文档...")
            ids_to_delete = []
            for relative_path, doc_file, current_hash in files_to_update:
                print(f"   🔄 更新: {relative_path}")
                results = self.collection.get(where={"source": relative_path}, include=[])
                ids_to_delete.extend(results['ids'])

            if ids_to_delete:
                delete_batch_size = 500
                for i in range(0, len(ids_to_delete), delete_batch_size):
                    self.collection.delete(ids=ids_to_delete[i:i + delete_batch_size])
                print(f"✅ 已删除 {len(ids_to_delete)} 个旧文档块")

        # ========== 第七步: 添加新文档和更新文档 ==========
        files_to_process = files_to_add + files_to_update

        if not files_to_process:
            print(f"\n📊 数据库当前共有 {self.collection.count()} 个文档块。")
            return

        print(f"\n💾 开始处理 {len(files_to_process)} 个文档...")

        new_docs_content = []
        new_docs_for_embedding = []
        new_ids = []
        new_metadatas = []

        for relative_path, doc_file, current_hash in files_to_process:
            action = "新增" if (relative_path, doc_file, current_hash) in files_to_add else "更新"
            print(f"\n📄 {action}: {relative_path}")

            content = self._read_file(doc_file)
            if content:
                chunks = self._smart_chunk_document(content, relative_path)
                print(f"   ✅ 生成 {len(chunks)} 个文档块")

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
            else:
                print(f"   ❌ 文件读取失败，跳过")

        # ========== 第八步: 生成嵌入并添加到数据库 ==========
        if new_docs_content:
            print(f"\n💾 共计 {len(new_docs_content)} 个新文档块待处理...")
            print("🔄 生成嵌入向量（可能需要一些时间）...")

            embedding_batch_size = 32
            embeddings = []
            for i in range(0, len(new_docs_for_embedding), embedding_batch_size):
                batch = new_docs_for_embedding[i:i + embedding_batch_size]
                batch_embeddings = self.embed_model.encode(batch, convert_to_tensor=False,
                                                           show_progress_bar=False).tolist()
                embeddings.extend(batch_embeddings)
                print(
                    f"   嵌入向量进度: {min(i + embedding_batch_size, len(new_docs_for_embedding))}/{len(new_docs_for_embedding)}")

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
                print(f"   批次 {i // db_batch_size + 1}/{total_batches} 添加成功")

            print(f"✅ 成功添加 {len(new_docs_content)} 个文档块")

        # ========== 最终统计 ==========
        print(f"\n✅ 增量更新完成!")
        print(f"📊 数据库当前共有 {self.collection.count()} 个文档块")

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