# 🎓 CQUPT AI 学生成长助手

> 第二届重庆市AI大模型创新应用大赛 · 超星泛雅集团出题赛道
>
> 基于 RAG 的高校学生成长与心理健康一站式咨询助手

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1.0-green)](https://fastapi.tiangolo.com)
[![Eval Pass Rate](https://img.shields.io/badge/eval-92.9%25-brightgreen)](data/eval_report.md)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## 🎯 一句话

一个能听懂你在问政策、在查数据、还是在求助的 RAG 智能助手。覆盖就业、考研、心理健康三大场景，回答前先过四道护栏。

---

## 🏗️ 系统架构

```
用户提问（Web / API）
        │
        ▼
┌─────────────────┐
│  意图路由        │  ← 关键词规则 + LLM 兜底 (100% 准确率)
│  policy/data/psy │
└────────┬────────┘
         │
    ┌────┼────┬──────────────┐
    ▼    ▼    ▼              ▼
  政策  数据  心理          高危检测
    │    │    │              │
    ▼    ▼    ▼              ▼
┌──────────────────────────────┐
│  混合检索引擎                 │
│  ┌───────┐  ┌──────┐        │
│  │Embedding│  │ BM25 │        │
│  │ (豆包)  │  │(jieba)│       │
│  └───┬───┘  └──┬───┘        │
│      └────┬────┘             │
│           ▼                  │
│      Top-K 文档              │
└──────────┬───────────────────┘
           ▼
┌─────────────────┐
│  LLM 生成回答    │  ← 豆包 API (doubao-seed-2-0-mini)
│  + 流式输出     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  四层护栏       │
│  危机/药物/免责/来源 │
└────────┬────────┘
         ▼
      返回用户
```

---

## 📊 评测结果

| 指标 | 数值 |
|---|---|
| **总通过率** | **92.9%** (13/14) |
| 意图路由准确率 | 100.0% (14/14) |
| 政策查询 | 100% (4/4) |
| 数据查询 | 100% (4/4) |
| 心理支持 | 67% (2/3) |
| 安全场景 | 100% (3/3) |
| 平均响应时间 | 29.8s |

评测框架：14 条测试用例覆盖 4 个场景 × 5 个维度（意图分类、关键词命中、禁用词拦截、来源标注、免责声明）

---

## 🚀 快速开始

### 1. 环境准备

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. 配置 API Key

```powershell
# Windows PowerShell
$env:DOUBAO_API_KEY = "你的火山引擎API Key"
```

或直接在 `backend/config.py` 中修改。

### 3. 知识库初始化

```bash
# 把竞赛提供的文档放入 data/documents/
# 然后运行：
python backend/test_pipeline.py
```

首次运行会调用豆包 Embedding API 对文档向量化，约 1-2 分钟。

### 4. 启动服务

```bash
python backend/main.py
```

浏览器打开 `http://localhost:8000`

---

## 📁 项目结构

```
cqupt-ai-assistant/
├── backend/
│   ├── main.py              # FastAPI 服务入口 + API 路由
│   ├── config.py            # 豆包 API、Embedding 模型配置
│   ├── answer_generator.py  # 回答生成 + EntityExtractor + HybridSearcher
│   ├── intent_router.py     # 意图分类（规则 + LLM 兜底）
│   ├── vector_store.py      # 混合检索 (Embedding + BM25 + 结构化查询)
│   ├── document_loader.py   # 多格式文档加载 (PDF/Word/Excel)
│   ├── guardrails.py        # 四层护栏 (危机/药物/免责/来源)
│   ├── eval_framework.py    # 评测框架 (14 条用例 × 5 维度)
│   ├── test_pipeline.py     # 端到端测试
│   └── requirements.txt
├── frontend/
│   └── index.html           # 单页面 Web 前端 (流式对话 + 场景切换)
├── data/
│   ├── documents/           # 原始文档存放
│   ├── store/               # 向量库持久化 (JSON)
│   └── eval_report.md       # 最新评测报告
├── run.bat                  # Windows 一键启动
└── README.md
```

---

## 🛡️ 安全机制

| 层级 | 检测内容 | 触发动作 |
|---|---|---|
| **输入层** | 危机关键词 (自杀/自残等) | 直接返回危机干预热线 |
| **生成层** | 诊断语句 / 药物推荐 | 拦截 + 替换为安全回复 |
| **输出层** | 心理回复缺少免责声明 | 自动追加 |
| **事实层** | 政策/数据缺乏来源标注 | 标注异常（非拦截） |

---

## 🔧 技术栈

| 组件 | 选型 |
|---|---|
| LLM | 豆包 (火山引擎 Ark) doubao-seed-2-0-mini |
| Embedding | 豆包 doubao-embedding-vision-251215 |
| 关键词检索 | BM25 + jieba 分词 |
| 文档解析 | pdfplumber + python-docx + openpyxl |
| Web 框架 | FastAPI + 流式 SSE |
| 前端 | 原生 HTML/CSS/JS (零框架) |
| 评测 | 自研 eval_framework (14 条多维度用例) |

---

## 📝 License

MIT

---

> 🏆 本项目为第二届重庆市AI大模型创新应用大赛参赛作品
>
> 作者：[朱郅 (泡芙)](https://github.com/zhu-zhi)
