# MyRag - 多模态 RAG 知识库系统

一个基于 LangChain 的多模态检索增强生成（RAG）知识库系统，支持文档解析、智能检索和多模态问答。

## 核心特性

### 多模态文档处理
- **多格式支持**：PDF、DOCX、Markdown、TXT、CSV
- **智能文档解析**：集成 MinerU 实现高质量 PDF 解析，自动提取文本、表格和图片
- **两阶段图片处理**：
  - 检索阶段：使用 VLM 生成图片描述，内联到文本中用于语义检索
  - 生成阶段：将原始图片传给多模态模型，生成高质量回答

### 灵活索引策略
| 索引类型 | 描述 | 适用场景 |
|---------|------|---------|
| 普通索引 | 标准分块向量索引 | 通用场景 |
| 摘要索引 | 文档摘要+全文存储 | 需要快速概览 |
| 父子索引 | 父文档+子块双粒度 | 需要精确定位和完整上下文 |

### 混合检索架构
- **向量检索**：基于 Chroma 的语义相似度检索
- **关键词检索**：BM25 实现精确关键词匹配
- **混合排序**：EnsembleRetriever 融合向量+关键词结果
- **智能重排序**：阿里百炼 Rerank 模型优化排序

## 项目结构

```
myRag/
├── custom_loader.py      # 自定义文档加载器（MinerU + VLM）
├── combine_client.py     # 知识库管理与检索生成核心
├── models.py             # LLM 和嵌入模型客户端
├── logger.py             # 日志配置
├── requirements.txt      # 项目依赖
├── chroma/               # 向量数据库目录
│   ├── knowledge/        # 上传的知识文档
│   └── _mineru_cache/    # MinerU 解析缓存
└── logs/                 # 日志文件目录
```

## 快速开始

### 环境配置

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置环境变量**
创建 `.env` 文件，配置以下参数：
```env
# 阿里云百炼（嵌入模型和重排序）
ALI_API_KEY=your_ali_api_key
ALI_TONGYI_EMBEDDING_MODEL=text-embedding-v3
ALI_TONGYI_RERANK_MODEL=rerank-v1

# LLM 服务（支持多种平台）
YUNWU_BASE_URL=https://api.yunwu.ai/v1
YUNWU_KEY=your_api_key

# Redis 存储
REDIS_URL=redis://localhost:6379
```

3. **启动 Redis**
```bash
# Windows
redis-server.exe

# Docker
docker run -d -p 6379:6379 redis:latest
```

### 基础使用

```python
from combine_client import MyKnowledge

# 1. 初始化知识库
kb = MyKnowledge()

# 2. 上传文档
MyKnowledge.upload_knowledge('path/to/your/document.pdf')

# 3. 加载知识库（创建索引）
kb.load_knowledge(index_type='parent_child')

# 4. 检索问答
result = kb.retrieve_and_answer(
    query='公司资产和负债占比多少',
    file_name='document.pdf',
    index_type='parent_child'
)

print(result['answer'])          # 回答内容
print(result['source_docs'])     # 引用的文档片段
print(result['image_count'])     # 涉及图片数量
```

### 高级用法

**自定义检索流程**
```python
# 获取检索器
retriever = kb.get_retriever('document.pdf', index_type='summary')

# 执行检索
docs = retriever.invoke('查询问题')

# 自定义生成
answer = kb.generate_answer('查询问题', docs)
```

**支持多模态生成（自动识别图片）**
```python
# 如果检索结果包含图片，系统会自动将原图传给 VLM
result = kb.retrieve_and_answer(
    query='分析这张图表的数据趋势',
    file_name='report_with_charts.pdf'
)
# 返回的答案会结合图片内容生成
```

## 核心模块说明

### custom_loader.py
自定义文档加载器，集成 MinerU 解析能力和 VLM 图片描述：
- `MyCustomLoader`：支持多格式文档的智能加载
- 自动处理文档中的图片，生成描述用于检索
- 构建图片注册表（图片描述 → 原图路径）

### combine_client.py
知识库管理与检索生成核心模块：
- `MyKnowledge`：知识库管理类
  - `upload_knowledge()`：上传文档
  - `load_knowledge()`：加载知识库（创建索引）
  - `get_retriever()`：获取检索器
  - `retrieve_and_answer()`：一站式检索问答
  - `generate_answer()`：多模态答案生成

### models.py
模型客户端封装：
- `get_lc_model_client()`：获取 LangChain 模型客户端
- `get_ali_embeddings()`：获取阿里嵌入模型
- `get_ali_rerank()`：获取阿里重排序模型

## 依赖说明

核心依赖项：
- **LangChain 生态**：`langchain-core`, `langchain-chroma`, `langchain-community`
- **文档解析**：`mineru`（PDF 解析）, `unstructured`
- **向量数据库**：`chromadb`
- **缓存存储**：`redis`
- **模型服务**：`langchain-openai`, `dashscope`

## 工作原理

### 文档处理流程
```
PDF/DOCX/MD → MinerU 解析 → 结构化内容(content_list)
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
            文本块               图片                   表格
              ↓                     ↓                     ↓
         直接索引          VLM 生成描述          HTML 转文本
              │              原图路径注册               │
              └─────────────────────┬─────────────────────┘
                                    ↓
                            统一文本 + 图片注册表
```

### 检索流程
```
用户查询
    ↓
┌─────────────┬─────────────┐
↓             ↓             ↓
向量检索    BM25检索     Rerank重排序
(Chroma)    (关键词)      (阿里百炼)
    │           │             │
    └───────────┴─────────────┘
              ↓
        检索结果文档
              ↓
    检查 metadata["image_paths"]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
 无图片              有图片
    ↓                   ↓
 文本生成          多模态生成
(普通LLM)          (VLM处理)
```

## 注意事项

1. **MinerU 模型下载**：首次使用会自动从 ModelScope 下载模型到 `./models` 目录
2. **内存占用**：PDF 解析需要较多内存，建议处理大文档时预留 4GB+ 内存
3. **Redis 依赖**：父子索引和摘要索引需要 Redis 作为文档存储后端
4. **API 费用**：VLM 图片描述和多模态生成会消耗额外的 API Token

## License

MIT License
