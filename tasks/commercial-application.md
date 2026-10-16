# 商业产品榜评测申请（草稿，10-18 前发出）

收件：contactus@agentmemoryleaderboard.ai
主题：Commercial Products Track Evaluation Request — Coding Track（AML 编程赛道）

## 待用户拍板
- [ ] 产品名（建议：Atlas Memory for Coding — Commercial Edition；与现有开源署名衔接）
- [ ] 发件身份/邮箱（公司名义 or 个人名义——商业榜不参与奖金，纯名次）
- [ ] 评测专用 token（部署 v1 后生成新 token，不复用 57.67 版的 22c7ad…）

## 邮件正文（英文）

Subject: Commercial Products Track — Evaluation Access Request (Coding)

Dear AML Team,

We would like to submit our commercial memory service for the **Commercial Products
Track, Coding Track (2nd cycle)**.

1. Product & version
   - Product: <产品名>
   - Version: v1.0 (frozen at Full acceptance; version policy: any change after Full
     acceptance is submitted as a new version)

2. API access (Add / Search, per the public contract)
   - Base endpoint: http://43.106.60.85:8002 (dedicated instance for evaluation,
     deployed alongside but isolated from our other service)
   - Auth: `Authorization: Token <EVAL_TOKEN>` (dedicated token for AML evaluation)
   - Endpoints: POST /add, POST /search, GET /health (no auth, for probing)
   - Contract compliance: synchronous Add (200 returned only after persistence and
     immediate searchability); Search returns stored memory records only — no answer
     generation at Search time; strict user_id isolation.

3. Capacity & limits
   - Max Add concurrency: 64; sustained: ~1,500 Add requests per run
   - Request duration: typical < 60s (Add), < 10s (Search); hard cap 30 min per request
   - Rate limiting: none from our side during evaluation windows
   - Burst behavior: 429 not expected; if hit, Retry-After honored
   - Daily cap: none; expected full-run completion: within 48h

4. Evaluation key
   Please issue a Leaderboard API Key for this commercial entry. Two clarifications
   we'd appreciate confirmation on:
   - Does this commercial entry receive its own Full quota (2 per track) independent
     of our existing open-source Methods key?
   - When does the 30-day cooldown for a second Full start counting?

Contact: <姓名 / 邮箱 / 时区>

Best regards,
<署名>

## 发送前检查
- [ ] v1 已部署到 :8002 且 /health 通过（或注明部署时间表）
- [ ] 新 token 已生成并配置
- [ ] smoke 一次通过后再正式发 full
