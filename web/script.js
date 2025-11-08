// API 配置
const API_BASE_URL = 'http://localhost:8000';

// 全局状态
let currentSessionId = null;
let isTyping = false;

// DOM 元素
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const reloadBtn = document.getElementById('reloadBtn');
const docCountEl = document.getElementById('docCount');
const chunkCountEl = document.getElementById('chunkCount');
const documentsListEl = document.getElementById('documentsList');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initializeMarked();
    initializeApp();
    setupEventListeners();
});

// 配置 Marked.js
function initializeMarked() {
    // 检查 marked 是否加载
    if (typeof marked === 'undefined') {
        console.error('❌ Marked.js 未加载！');
        showToast('Markdown 库加载失败', 'error');
        return;
    }

    // 检查 hljs 是否加载
    if (typeof hljs === 'undefined') {
        console.warn('⚠️ Highlight.js 未加载，代码高亮将不可用');
    }

    // 配置 marked 选项
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false,
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (err) {
                    console.warn('代码高亮失败:', err);
                }
            }
            return typeof hljs !== 'undefined' ? hljs.highlightAuto(code).value : code;
        }
    });

    console.log('✅ Markdown 渲染器初始化成功');
}

// 初始化应用
async function initializeApp() {
    try {
        // 检查健康状态
        const health = await fetchAPI('/api/health');
        if (!health.status) {
            throw new Error('API 服务未响应');
        }

        // 获取系统信息
        await loadSystemInfo();

        // 创建新会话
        await createNewSession();

        showToast('系统初始化成功', 'success');
    } catch (error) {
        console.error('初始化失败:', error);
        showToast('无法连接到服务器,请确保后端服务已启动', 'error');
    }
}

// 设置事件监听器
function setupEventListeners() {
    // 发送消息
    sendBtn.addEventListener('click', handleSendMessage);

    // 回车发送,Shift+回车换行
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // 自动调整输入框高度
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    });

    // 新对话
    newChatBtn.addEventListener('click', async () => {
        if (confirm('确定要开始新对话吗?当前对话历史将被清除。')) {
            await createNewSession();
            clearChat();
            showToast('已开始新对话', 'success');
        }
    });

    // 重载文档
    reloadBtn.addEventListener('click', async () => {
        if (confirm('确定要重新加载文档吗?这可能需要一些时间。')) {
            await reloadDocuments();
        }
    });
}

// API 请求封装
async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: '请求失败' }));
        throw new Error(error.detail || '请求失败');
    }

    return response.json();
}

// 加载系统信息
async function loadSystemInfo() {
    try {
        const info = await fetchAPI('/api/info');

        docCountEl.textContent = info.total_documents || 0;
        chunkCountEl.textContent = info.total_chunks || 0;

        // 显示文档列表
        if (info.documents && info.documents.length > 0) {
            documentsListEl.innerHTML = info.documents
                .map(doc => `
                    <div class="doc-item">
                        <div class="doc-name" title="${doc.source}">${doc.source}</div>
                        <div class="doc-meta">
                            <span>${doc.chunk_count} 块</span>
                            <span>${formatBytes(doc.file_size)}</span>
                        </div>
                    </div>
                `).join('');
        } else {
            documentsListEl.innerHTML = '<div class="loading">暂无文档</div>';
        }
    } catch (error) {
        console.error('加载系统信息失败:', error);
        documentsListEl.innerHTML = '<div class="loading">加载失败</div>';
    }
}

// 创建新会话
async function createNewSession() {
    try {
        const response = await fetchAPI('/api/session/create', {
            method: 'POST'
        });
        currentSessionId = response.session_id;
        console.log('创建新会话:', currentSessionId);
    } catch (error) {
        console.error('创建会话失败:', error);
        showToast('创建会话失败', 'error');
    }
}

// 发送消息
async function handleSendMessage() {
    const query = messageInput.value.trim();

    if (!query || isTyping) return;

    // 添加用户消息
    addMessage('user', query);

    // 清空输入框
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // 设置发送状态
    isTyping = true;
    sendBtn.disabled = true;

    try {
        await streamAnswer(query);
    } catch (error) {
        console.error('发送消息失败:', error);
        addMessage('assistant', '抱歉,发生了错误: ' + error.message);
        showToast('发送失败', 'error');
    } finally {
        isTyping = false;
        sendBtn.disabled = false;
    }
}

// 流式获取答案
async function streamAnswer(query) {
    // 显示打字指示器
    const typingId = addTypingIndicator();

    // 创建助手消息容器
    const messageEl = createMessageElement('assistant');
    const bubbleEl = messageEl.querySelector('.message-bubble');
    bubbleEl.textContent = '';

    let fullAnswer = '';
    let sources = [];

    try {
        const response = await fetch(`${API_BASE_URL}/api/ask/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) {
            throw new Error('请求失败');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        // 移除打字指示器并添加消息元素
        removeTypingIndicator(typingId);
        chatMessages.appendChild(messageEl);

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n\n');

            for (const line of lines) {
                if (!line.trim() || !line.startsWith('data: ')) continue;

                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.type === 'session') {
                        currentSessionId = data.session_id;
                    } else if (data.type === 'sources') {
                        sources = data.content;
                    } else if (data.type === 'content') {
                        fullAnswer += data.content;

                        // 实时渲染 Markdown（每次都渲染以保证显示）
                        if (typeof marked !== 'undefined') {
                            try {
                                bubbleEl.innerHTML = marked.parse(fullAnswer);
                            } catch (e) {
                                console.warn('Markdown 解析警告:', e);
                                bubbleEl.textContent = fullAnswer;
                            }
                        } else {
                            // marked 未加载，使用纯文本
                            bubbleEl.textContent = fullAnswer;
                        }
                        scrollToBottom();

                    } else if (data.type === 'done') {
                        // 最终渲染
                        if (typeof marked !== 'undefined') {
                            try {
                                bubbleEl.innerHTML = marked.parse(fullAnswer);
                                console.log('✅ 最终 Markdown 渲染完成');
                            } catch (e) {
                                console.error('❌ 最终 Markdown 渲染失败:', e);
                                bubbleEl.textContent = fullAnswer;
                            }
                        } else {
                            bubbleEl.textContent = fullAnswer;
                        }

                        // 添加来源信息
                        if (sources.length > 0) {
                            addSourcesInfo(messageEl, sources);
                        }
                        scrollToBottom();
                    } else if (data.type === 'error') {
                        bubbleEl.textContent = '❌ ' + data.content;
                    }
                } catch (e) {
                    console.error('解析数据失败:', e);
                }
            }
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        throw error;
    }
}

// 添加消息
function addMessage(role, content, isMarkdown = true) {
    const messageEl = createMessageElement(role);
    const bubbleEl = messageEl.querySelector('.message-bubble');

    // 助手消息始终尝试渲染 Markdown，用户消息保持纯文本
    if (role === 'assistant' && isMarkdown && typeof marked !== 'undefined') {
        try {
            bubbleEl.innerHTML = marked.parse(content);
            console.log('✅ Markdown 渲染成功');
        } catch (e) {
            console.error('❌ Markdown 渲染失败:', e);
            bubbleEl.textContent = content;
        }
    } else {
        bubbleEl.textContent = content;
    }

    chatMessages.appendChild(messageEl);
    scrollToBottom();
    return messageEl;
}

// 创建消息元素
function createMessageElement(role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '我' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    contentDiv.appendChild(bubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);

    return messageDiv;
}

// 添加来源信息
function addSourcesInfo(messageEl, sources) {
    const contentDiv = messageEl.querySelector('.message-content');

    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'message-sources';

    const header = document.createElement('div');
    header.className = 'sources-header';
    header.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 2V14M2 8H14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        参考来源 (${sources.length})
    `;

    sourcesDiv.appendChild(header);

    sources.slice(0, 5).forEach(source => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';

        const keywords = source.keywords && source.keywords.length > 0
            ? source.keywords.slice(0, 3).map(kw => `<span class="source-tag">${kw}</span>`).join(' ')
            : '';

        sourceItem.innerHTML = `
            <div class="source-name">${source.source}</div>
            <div class="source-meta">
                ${source.section_path ? `<span>📍 ${source.section_path}</span>` : ''}
                <span>🎯 ${source.semantic_similarity}%</span>
                ${source.keyword_score > 0 ? `<span>🔑 ${source.keyword_score}%</span>` : ''}
            </div>
            ${keywords ? `<div class="source-meta" style="margin-top: 4px">${keywords}</div>` : ''}
        `;

        sourcesDiv.appendChild(sourceItem);
    });

    contentDiv.appendChild(sourcesDiv);
}

// 添加打字指示器
function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const messageEl = createMessageElement('assistant');
    messageEl.id = id;

    const bubbleEl = messageEl.querySelector('.message-bubble');
    bubbleEl.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;

    chatMessages.appendChild(messageEl);
    scrollToBottom();

    return id;
}

// 移除打字指示器
function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// 清空聊天
function clearChat() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                    <circle cx="32" cy="32" r="28" stroke="url(#welcomeGradient)" stroke-width="3"/>
                    <path d="M24 32L30 38L40 26" stroke="url(#welcomeGradient)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                    <defs>
                        <linearGradient id="welcomeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#667eea"/>
                            <stop offset="100%" style="stop-color:#764ba2"/>
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <h2>欢迎使用 RAG 智能问答系统</h2>
            <p>基于混合检索策略，结合关键词匹配和语义理解</p>
            <div class="welcome-features">
                <div class="feature">
                    <span class="feature-icon">🎯</span>
                    <span>精准检索</span>
                </div>
                <div class="feature">
                    <span class="feature-icon">💡</span>
                    <span>智能问答</span>
                </div>
                <div class="feature">
                    <span class="feature-icon">📚</span>
                    <span>多轮对话</span>
                </div>
            </div>
        </div>
    `;
}

// 重载文档
async function reloadDocuments() {
    const originalText = reloadBtn.textContent;
    reloadBtn.disabled = true;
    reloadBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" class="spinning">
            <path d="M2 8C2 4.686 4.686 2 8 2C11.314 2 14 4.686 14 8C14 11.314 11.314 14 8 14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        重载中...
    `;

    try {
        await fetchAPI('/api/reload', {
            method: 'POST',
            body: JSON.stringify({ force_reload: true })
        });

        await loadSystemInfo();
        showToast('文档重载成功', 'success');
    } catch (error) {
        console.error('重载文档失败:', error);
        showToast('重载失败: ' + error.message, 'error');
    } finally {
        reloadBtn.disabled = false;
        reloadBtn.textContent = originalText;
    }
}

// 滚动到底部
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 显示 Toast
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// 格式化字节
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 添加旋转动画
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .spinning {
        animation: spin 1s linear infinite;
    }
`;
document.head.appendChild(style);