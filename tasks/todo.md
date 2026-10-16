# memory_coding v1 开发清单（商业榜冲顶）

## Phase A 核心模块
- [x] A1 `app/llm.py` OpenRouter 统一客户端（重试/超时/JSON 提取 + base-url/.env 支持）
- [x] A2 `app/distill.py` Add 侧蒸馏（Pass1 生成 + 事实校验 + profile 卡）
- [x] A3 `app/store.py` 扩展：kind 列/卡双索引/卡加权/邻接守卫/search_recall 拆分
- [x] A4 `app/judge.py` Search 侧判别 + 题型条件化打包 + 预算裁剪
- [x] A5 `app/main.py` 接线（DISTILL/JUDGE 环境开关，全程降级路径）
- [x] A6 冒烟测试（卡加权生效、bug/feature 打包分支、摘录配对全部通过 2026-10-16）

## Phase B 离线验证（PoC）
- [ ] B1 `scripts/eval_answer_sim.py` answer 仿真闭环（localization 硬指标 + LLM 软判分，docker 后端留服务器）
- [ ] B2 smoke 子集（8 题）验证管道
- [ ] B3 PoC 正式跑：~24-48 题 × relevant/noisy × baseline vs v1 配对
- [ ] B4 结果报告 + 对照 G 门禁

## Phase C 生产化
- [ ] C1 部署阿里云（新容器/端口，与 57.67 版并行）
- [ ] C2 商业榜申请邮件发出（10-18 前）
- [ ] C3 全量消融 + G1-G4 门禁
- [ ] C4 smoke → full

## 里程碑
- spec: docs/superpowers/specs/2026-10-16-commercial-track-v1-design.md
- 基线（新仓库 Lite）: HitRate@100=96.08% MRR=0.683 (2026-10-16)
