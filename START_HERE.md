# 两阶段作业导航

更新：2026-10-10。本次仅整理文档，尚未开始 Stage 2 实验。

**Stage 2 的启动资料已经齐全，可以据此开始实现、验证和提交。**
环境安装、框架实现和实验是后续工作，不是缺失的题目材料。

## 阅读入口

| 目的 | 入口 |
| --- | --- |
| 开始 Stage 2，了解做法、运行与提交 | [Stage 2 启动指南](stage2/START_HERE.md) |
| Stage 2 学习要求 | [Background](stage2/BACKGROUND.md) |
| Stage 2 权威技术与提交规则 | [Assignment Statement](stage2/README.md) |
| Stage 1 成果与可复用证据 | [Stage 1 状态](stage1/project/STAGE1_STATUS.md) |
| Stage 1 原始题目 | [原始 README](stage1/README.md) |
| 后续 agent 的阶段边界 | [AGENTS](AGENTS.md) |

## 目录现状

两阶段已完整拆分：

```text
仓库根目录/
├── .git/、.gitignore       # 公共版本管理
├── README.md、START_HERE.md、AGENTS.md
├── stage1/                 # 原作业、代码、环境、实验、历史和提交包
└── stage2/                 # 新作业 starter 与后续实现
```

Stage 1 命令从 `stage1/` 执行；Stage 2 命令从 `stage2/` 执行。
原题目、历史实验记录和已有提交包内容保持不变。历史记录中的绝对路径保留
作为当时的证据；复现时将旧仓库根路径映射到当前 `stage1/`。
ZIP 内部仍采用各自提交契约的根路径，不额外套 `stage1/` 或 `stage2/`。

## 不要混用的规则

| 项目 | Stage 1 | Stage 2 |
| --- | --- | --- |
| 程序 | 两个 ASM | 五个静态 JSONL，共用同一硬件 |
| 评测器 | Python/NumPy | Rust 核心、Python 调度 |
| MIP | 可选，已有部分实践 | 必须有实质建模与求解 |
| 多智能体框架 | 原有搜索过程 | 明确要求可运行框架与协作证据 |
| 面积 | ≤24 mm² | ≤100 AU＝25 mm² |
| 功率 | 1000 cycles ≤20 W | 100 cycles ≤34 W；10000 cycles ≤26 W |
| ZIP | ≤25 MiB；最终打包另检查解压 ≤100 MiB | ≤256 MiB；解压 ≤1 GiB；≤10000 条目 |
| 成绩 | cand-155 已有本地验证 | 尚未运行个人方案；旧分数不能直接沿用 |

Stage 1 文档中的 milestone 1/2 日期不等于两套评测器的技术契约。
Stage 2 文档截止时间为 **2026-10-12 北京时间 12:00**，使用独立提交入口。

## 执行时处理的事项

- 准备 Rust 1.90+；Python 要求 3.11+。审阅时 Python 为 3.12.3，
  当前 PATH 未找到 Cargo/Rust；尚未构建或测试。
- 先跑小例子和完整参考基线，再建立模型与优化框架。
- 提交网站登录、可达性和最新公告尚未在线核验。
- 旧用户指令只收集当前主会话，Stage 2 官方要求相关子会话等记录。
  开始相关流程前统一策略；保留既有历史，不擅自扩大收集范围。
