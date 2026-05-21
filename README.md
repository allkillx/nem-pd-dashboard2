# NEM PD Dashboard — GitHub Pages 部署版

完全免费、零运维的电力市场仪表盘。GitHub Actions 每 30 分钟自动拉 OpenElectricity 数据，GitHub Pages 托管前端。

## 🚀 部署步骤（首次设置一次，之后全自动）

### 第 1 步：清理旧 repo，推上新文件

你现有的 repo (`allkillx/nem-pd-dashboard`) 里有 Flask 框架残留 (`app.py`/`static/`/`templates/`)，**和这套方案冲突**。最简单的办法是清空重来：

```bash
# 在本地把这个 repo 重新拉一遍
git clone https://github.com/allkillx/nem-pd-dashboard.git
cd nem-pd-dashboard

# 删掉旧文件
rm -rf app.py static templates nem_dashboard
git rm -rf app.py static templates nem_dashboard 2>/dev/null || true

# 把这次给你的 4 个文件复制进去（index.html, fetch_nem.py, requirements.txt, .gitignore）
# 再把 .github/workflows/refresh.yml 和 data/meta.json 也复制进去
# （保留目录结构）

# 提交
git add .
git commit -m "rebuild: GitHub Pages + Actions architecture"
git push
```

最终目录结构应该是：
```
nem-pd-dashboard/
├── .github/
│   └── workflows/
│       └── refresh.yml          ← 每 30 分钟自动跑
├── data/
│   └── meta.json                ← 种子文件
├── .gitignore
├── fetch_nem.py                 ← 数据管线
├── index.html                   ← Dashboard（GitHub Pages 默认入口）
├── requirements.txt
└── README.md
```

### 第 2 步：加 OpenElectricity API key 到 Secrets

GitHub repo 页面 → **Settings** → 左侧 **Secrets and variables** → **Actions** → **New repository secret**

- Name: `OPENELECTRICITY_API_KEY`
- Value: 你从 https://platform.openelectricity.org.au/ 拿到的 key

### 第 3 步：开 GitHub Pages

repo 页面 → **Settings** → 左侧 **Pages**:
- **Source**: Deploy from a branch
- **Branch**: `main` / `/ (root)` → Save

等 1-2 分钟，会出现一个绿色提示带 URL，类似：
`https://allkillx.github.io/nem-pd-dashboard/`

打开就能看到 dashboard 了。这时还是合成数据，因为 Actions 还没跑过。

### 第 4 步：手动触发第一次数据拉取

repo 页面 → **Actions** 标签 → 左侧 **Refresh NEM data** → 右上 **Run workflow** 按钮 → 选 `backfill`（拉 30 天历史）→ Run。

跑 1-2 分钟，看到绿色 ✓ 后刷新 dashboard 页面 — 状态从 "Demo Sim" 变成 "Live"，数据切到真实的。

之后每 30 分钟自动跑 `full` 模式，啥也不用管了。

---

## 🔧 验证 / 排错

### 看 Action 跑得怎么样
**Actions** 标签 → 点开任意一次 run → 看 log。常见问题：

| 错误 | 原因 | 解决 |
|---|---|---|
| `OPENELECTRICITY_API_KEY env var not set` | secret 没设 | 回第 2 步 |
| `403 / 401 Unauthorized` | API key 错或没激活 | 去 platform.openelectricity.org.au 检查 |
| `No data returned from OpenElectricity API` | API 可能临时挂了 | 等下次 cron 重试 |
| `Permission denied to github-actions[bot]` | workflow 没 write 权限 | Settings → Actions → General → Workflow permissions → 选 **Read and write** |

### Dashboard 一直显示 Demo Sim
说明 dashboard 没拉到 `./data/*.json`。打开浏览器 DevTools (F12) → Console 标签，会看到：
```
[NEM] Live data unavailable, using synthetic fallback: ...
```
那条 `...` 就是原因。99% 是 `data/history_30d.json` 还不存在 — 去 Actions 标签手动跑一次。

### GitHub Pages URL 打开 404
- 确认 Settings → Pages 显示绿色 "Your site is live at..."
- 首次启用要等 2-5 分钟
- 文件名必须是 `index.html`（小写），不是 `Index.html` 或 `INDEX.html`

---

## 💰 成本

- GitHub Pages：**免费**（公开 repo 无限流量）
- GitHub Actions：**免费**（公开 repo 完全免费；私有 repo 每月 2000 分钟，这个工作流每月用 ~30 分钟）
- OpenElectricity API：**免费**（看他们当前的 rate limit 政策）

总成本：**0 澳元**。

---

## 🛠 自定义

### 改刷新频率
编辑 `.github/workflows/refresh.yml`：
```yaml
- cron: '*/30 * * * *'   # 30 分钟 → 改成 '*/15 * * * *' 是 15 分钟
```
注意：GitHub 免费 cron 最快是 5 分钟一次，且**繁忙时可能延迟 15 分钟以上**（这是 GitHub 的限制）。要严格 5min 实时性得自己跑 VPS。

### 改区域 / 时间范围
`fetch_nem.py` 里的 `REGIONS` 列表和 `days_map`。

### 升级模型
`fetch_nem.py` 里的 `forecast_region()` 函数。换成 LightGBM / XGBoost / Prophet / NeuralProphet 都行，只要输出的 JSON schema 不变（`timestamp / pred / lo / hi`），dashboard 一行不用改。

---

## 📊 Dashboard 功能速览

- **顶部 ticker**：5 个 region 实时 RRP + Δ%
- **主图**：30-min 价格 / 需求 / 可用发电三轴叠加，附 ML 预测虚线 + 历史 p10-p90 分布带
- **热力图**：14 天 × 48 间隔，颜色梯度看一眼知道每天哪个时段最贵
- **预测面板**：未来 24 个 interval（12 小时）+ 80% 置信区间
- **异常检测**：robust MAD z-score > 2.5 的尖峰/负价事件
- **数据表**：最近 48 个 interval 明细，带 7 日同时段对比和储备率告警
