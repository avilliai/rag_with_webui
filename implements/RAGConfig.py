from typing import Optional


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
