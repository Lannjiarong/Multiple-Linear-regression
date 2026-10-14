# 多元線性回歸 (Multiple Linear Regression) 實戰分析與 CRISP-DM 全流程實踐

- **學生學號：** 7115056142
- **分析資料集：** Kaggle Playground Series s4e5: [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5) (20 特徵，符合 10~20 特徵要求)
- **對照驗證集：** 50_Startups 獲利預測（完全還原作業參考圖片之特徵選擇曲線與表格）
- **Web 系統風格：** 參考國際頂級統計工具 [StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)

---

## 專案亮點與作業成果對照

1. **CRISP-DM 六大流程完整落實：**
   - **Business Understanding：** 水文防汛與災害風險量化。
   - **Data Understanding：** 20 個特徵相關性矩陣、敘述統計與視覺化。
   - **Data Preparation：** 80/20 Train/Test 切分與矩陣擴充。
   - **Modeling：** 逐步向前特徵選擇 (SFS)、最小平方法 (OLS)。
   - **Evaluation：** 五大回歸假設診斷、95% 信賴區間 (CI) 與 95% 預測區間 (PI) 預測圖。
   - **Deployment：** 打造現代 StatsUnlock 風格之 Web 互動計算機。

2. **作業參考圖片完全還原：**
   - `figures/rmse_r2_by_features.png`：特徵數量 (1~20) 對 RMSE 與 R-squared 變化之雙折線圖（對應參考圖一）。
   - `figures/feature_selection_table.png`：特徵選擇歷程評估表（對應參考圖二）。
   - `figures/50_startups_rmse_r2.png` & `50_startups_table.png`：50_Startups 之 1:1 對照圖表。

3. **StatsUnlock 風格 Web 應用 (`web/`)：**
   - 雙模式支援：可直接雙擊 `web/index.html` 離線在任意瀏覽器中執行（內建純 JS 矩陣 OLS 引擎與 Chart.js）；亦可透過 `python app.py` 啟動 FastAPI 伺服端服務（網址：http://127.0.0.1:8000）。
   - 功能包含：Model Summary 統計卡片、ANOVA 表、係數表 (含 95% CI & VIF)、雙折線特徵選擇歷程、95% CI/PI 區間帶圖、動態滑桿即時情境預測計算機、自動生成 APA 7th 統計報告。

4. **AI 輔助與文獻探討：**
   - **NotebookLM 研究摘要（100字以上）：** 探討 Kaggle S4E5 線上主流解法（列聚合特徵工程 + Ridge / LightGBM / CatBoost），並於報告中明確標示。
   - **GPT 輔助內容與對話匯出：** 完整記錄 AI 協作脈絡，並產出 `GPT_dialogue.pdf`。

---

## 繳交檔案清單

| 檔案名稱 | 說明 |
| :--- | :--- |
| **`7115056142_hw2.ipynb`** | Jupyter Notebook 主程式（含完整代碼、執行輸出、公式與圖表） |
| **`7115056142_hw2.py`** | Python 獨立主程式（自動執行資料處理、特徵選擇、圖表生成） |
| **`7115056142_hw2_report.pdf`** | 依照 CRISP-DM 規範撰寫之正式 PDF 評估報告書 |
| **`7115056142_hw2_report.md`** | 書面報告之原始 Markdown 文件 |
| **`GPT_dialogue.pdf`** | AI (GPT) 協作對話過程匯出紀錄 |
| **`7115056142_hw2.zip`** | 依照學號命名之全套作業封裝壓縮包 |
| **`app.py`** | FastAPI 後端伺服器 (啟動 Web 服務) |
| **`web/`** | StatsUnlock 風格多元線性回歸計算機前端套件 |
| **`figures/`** | 包含雙折線特徵選擇圖、特徵挑選表、CI/PI 圖、五大假設診斷圖等全部高解析度圖表 |
| **`data/`** | Kaggle 20 特徵洪水預測資料集與 50_Startups 對照資料集 |

---

## 執行方式

### 1. 執行 Python 分析主程式
```bash
python 7115056142_hw2.py
```

### 2. 啟動 Web 計算機系統
方式一：直接在檔案總管雙擊開啟 `web/index.html` 即可在瀏覽器中離線運行。  
方式二：啟動 FastAPI 本機伺服器：
```bash
python app.py
```
開啟瀏覽器訪問 `http://127.0.0.1:8000` 即可體驗完整功能。
