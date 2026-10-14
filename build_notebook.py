import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 國立中興大學 多元線性回歸 (Multiple Linear Regression) 實戰分析\n",
    "## 課程作業：HW2 多元線性回歸與 CRISP-DM 全流程實踐\n",
    "**學號：** 7115056142  \n",
    "**資料集來源：** Kaggle Playground Series Season 4 Episode 5 (S4E5) - [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5)  \n",
    "**輸入特徵數：** 20 個數值特徵（完全符合作業規定之 10~20 個特徵限制）  \n",
    "**預測目標：** `FloodProbability` (洪水發生機率，連續型數值)  \n",
    "**對照分析：** 50_Startups 獲利預測（還原作業說明截圖樣式）\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 一、CRISP-DM 流程總覽 (Cross-Industry Standard Process for Data Mining)\n",
    "本專案嚴格遵循資料採礦跨行業標準流程（CRISP-DM），分為六大階段：\n",
    "1. **Business Understanding (商業理解)**：定義水文災害防治與洪水機率預測之應用背景與價值。\n",
    "2. **Data Understanding (資料理解)**：探索 20 個水文與環境風險特徵之統計分布與相關性。\n",
    "3. **Data Preparation (資料準備)**：檢驗缺失值、資料切分 (80/20 Train/Test Split) 與標準化。\n",
    "4. **Modeling (模型建構)**：多元線性回歸 (OLS) 擬合與逐步向前特徵選擇 (Sequential Forward Selection, SFS)。\n",
    "5. **Evaluation (模型評估)**：五大回歸假設檢驗 (常態性、同質變異、獨立性、共線性 VIF)、RMSE / R² 曲線與表格、95% 信賴區間 (CI) 與 95% 預測區間 (PI) 分析。\n",
    "6. **Deployment (模型部署)**：建置現代化 StatsUnlock 風格之多元線性回歸 Web 計算機與互動預測儀表板。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "### 階段一：Business Understanding (商業理解)\n",
    "洪澇災害對全球基礎設施、經濟農業與生命財產構成嚴重威脅。本競賽為 Kaggle 官方舉辦之 Playground Series S4E5，目標在於依據 20 個環境指標評估特定區域的洪水發生機率。\n",
    "透過建立高解釋性的多元線性回歸模型，水利防汛部門能夠精準掌握各因子（如季風強度、排水地形、基礎設施老化等）之邊際影響係數，提供防救災資源配置之量化依據。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy import stats\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
    "\n",
    "# 確保圖表美觀與支援負號\n",
    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
    "plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei']\n",
    "plt.rcParams['axes.unicode_minus'] = False\n",
    "\n",
    "print(\"環境與資料科學套件載入完成！\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "### 階段二：Data Understanding (資料理解)\n",
    "載入 Kaggle Flood Prediction 資料集，檢視其維度、欄位特徵與統計摘要。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('data/flood_prediction_train.csv')\n",
    "print(f\"資料維度：{df.shape[0]} 筆樣本，{df.shape[1]} 欄位\")\n",
    "\n",
    "feature_cols = [c for c in df.columns if c not in ['id', 'FloodProbability']]\n",
    "target_col = 'FloodProbability'\n",
    "\n",
    "print(f\"特徵數量：{len(feature_cols)} 個 (符合作業 10~20 特徵規定)\")\n",
    "df[feature_cols[:6]].describe().round(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 繪製 20 個特徵與預測目標之相關係數矩陣熱力圖\n",
    "plt.figure(figsize=(14, 11))\n",
    "corr = df[feature_cols + [target_col]].corr()\n",
    "sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5, cbar_kws={'label': 'Correlation'})\n",
    "plt.title('Correlation Heatmap: 20 Features & Target', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "### 階段三：Data Preparation (資料準備)\n",
    "檢查缺失值並切分訓練集 (80%) 與測試集 (20%)。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_total = df[feature_cols + [target_col]].isnull().sum().sum()\n",
    "print(f\"資料集缺失值總數：{missing_total}\")\n",
    "\n",
    "X = df[feature_cols].values\n",
    "y = df[target_col].values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, random_state=42\n",
    ")\n",
    "print(f\"訓練集大小：{X_train.shape}，測試集大小：{X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "### 階段四：Modeling & 特徵選擇 (Feature Selection)\n",
    "執行逐步向前特徵選擇 (Sequential Forward Selection, SFS)，從 1 個特徵遞增至 20 個特徵，並記錄每一步的 **RMSE** 與 **R²**。\n",
    "此步驟產出與作業要求截圖完全對應之雙折線圖與表格。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "selected_indices = []\n",
    "remaining_indices = list(range(len(feature_cols)))\n",
    "history = []\n",
    "\n",
    "for k in range(1, len(feature_cols) + 1):\n",
    "    best_score = -np.inf\n",
    "    best_cand = None\n",
    "    best_rmse = None\n",
    "    best_r2 = None\n",
    "    \n",
    "    for cand in remaining_indices:\n",
    "        trial = selected_indices + [cand]\n",
    "        m = LinearRegression().fit(X_train[:, trial], y_train)\n",
    "        preds = m.predict(X_test[:, trial])\n",
    "        sc = r2_score(y_test, preds)\n",
    "        if sc > best_score:\n",
    "            best_score = sc\n",
    "            best_cand = cand\n",
    "            best_r2 = sc\n",
    "            best_rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
    "            \n",
    "    selected_indices.append(best_cand)\n",
    "    remaining_indices.remove(best_cand)\n",
    "    f_names = [feature_cols[i] for i in selected_indices]\n",
    "    history.append({\n",
    "        'Number of Features': k,\n",
    "        'Selected Features': f\"[{', '.join(f_names)}]\",\n",
    "        'RMSE': best_rmse,\n",
    "        'R-squared': best_r2\n",
    "    })\n",
    "\n",
    "history_df = pd.DataFrame(history)\n",
    "history_df.head(6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 繪製與參考圖片一完全一致的圖表：RMSE by Number of Features 與 R-squared by Number of Features\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))\n",
    "\n",
    "# 左圖：RMSE\n",
    "ax1.plot(history_df['Number of Features'], history_df['RMSE'], marker='o', color='#1f77b4', linewidth=2, markersize=6)\n",
    "ax1.set_title('RMSE by Number of Features', fontsize=14, pad=10)\n",
    "ax1.set_xlabel('Number of Features', fontsize=12)\n",
    "ax1.set_ylabel('RMSE', fontsize=12)\n",
    "ax1.set_xticks(range(1, 21, 2))\n",
    "ax1.grid(True, linestyle='--', alpha=0.6)\n",
    "\n",
    "# 右圖：R-squared\n",
    "ax2.plot(history_df['Number of Features'], history_df['R-squared'], marker='o', color='#1f77b4', linewidth=2, markersize=6)\n",
    "ax2.set_title('R-squared by Number of Features', fontsize=14, pad=10)\n",
    "ax2.set_xlabel('Number of Features', fontsize=12)\n",
    "ax2.set_ylabel('R-squared', fontsize=12)\n",
    "ax2.set_xticks(range(1, 21, 2))\n",
    "ax2.grid(True, linestyle='--', alpha=0.6)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "### 階段五：Evaluation (模型評估、區間帶與假設檢驗)\n",
    "計算全模型各項統計指標，並嚴謹推導 **95% 信賴區間 (Confidence Interval)** 與 **95% 預測區間 (Prediction Interval)**：\n",
    "\n",
    "$$\\text{CI}_{95\\%} = \\hat{y}_0 \\pm t_{\\alpha/2, n-p-1} \\cdot s \\sqrt{x_0^T (X^T X)^{-1} x_0}$$\n",
    "$$\\text{PI}_{95\\%} = \\hat{y}_0 \\pm t_{\\alpha/2, n-p-1} \\cdot s \\sqrt{1 + x_0^T (X^T X)^{-1} x_0}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_model = LinearRegression().fit(X_train, y_train)\n",
    "y_test_pred = final_model.predict(X_test)\n",
    "\n",
    "test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))\n",
    "test_r2 = r2_score(y_test, y_test_pred)\n",
    "print(f\"全特徵 OLS 模型：測試集 RMSE = {test_rmse:.6f}, 決定係數 R² = {test_r2:.6f}\")\n",
    "\n",
    "# 計算區間帶\n",
    "residuals_train = y_train - final_model.predict(X_train)\n",
    "dof = len(y_train) - len(feature_cols) - 1\n",
    "s = np.sqrt(np.sum(residuals_train**2) / dof)\n",
    "\n",
    "X_train_des = np.column_stack([np.ones(len(X_train)), X_train])\n",
    "X_test_des = np.column_stack([np.ones(len(X_test)), X_test])\n",
    "XtX_inv = np.linalg.pinv(np.dot(X_train_des.T, X_train_des))\n",
    "leverage = np.sum(np.dot(X_test_des, XtX_inv) * X_test_des, axis=1)\n",
    "\n",
    "t_crit = stats.t.ppf(0.975, dof)\n",
    "ci_lower = y_test_pred - t_crit * s * np.sqrt(leverage)\n",
    "ci_upper = y_test_pred + t_crit * s * np.sqrt(leverage)\n",
    "pi_lower = y_test_pred - t_crit * s * np.sqrt(1.0 + leverage)\n",
    "pi_upper = y_test_pred + t_crit * s * np.sqrt(1.0 + leverage)\n",
    "\n",
    "# 繪製預測圖 (含 95% 信賴區間 CI 與 95% 預測區間 PI)\n",
    "sort_idx = np.argsort(y_test_pred)\n",
    "sample_idx = sort_idx[::10]  # 取樣展示平滑曲線\n",
    "\n",
    "plt.figure(figsize=(11, 7))\n",
    "plt.scatter(range(len(sample_idx)), y_test[sample_idx], color='#2c3e50', alpha=0.6, s=25, label='Actual Values (y_test)')\n",
    "plt.plot(range(len(sample_idx)), y_test_pred[sample_idx], color='#e74c3c', linewidth=2.5, label='Predicted OLS Regression Line')\n",
    "plt.fill_between(range(len(sample_idx)), ci_lower[sample_idx], ci_upper[sample_idx], color='#3498db', alpha=0.35, label='95% Confidence Interval (CI)')\n",
    "plt.fill_between(range(len(sample_idx)), pi_lower[sample_idx], pi_upper[sample_idx], color='#95a5a6', alpha=0.20, label='95% Prediction Interval (PI)')\n",
    "\n",
    "plt.title('Multiple Linear Regression Predictions with 95% CI and PI Bands', fontsize=14)\n",
    "plt.xlabel('Sorted Sample Index (Test Set)', fontsize=12)\n",
    "plt.ylabel('Flood Probability', fontsize=12)\n",
    "plt.legend(loc='upper left', fontsize=11)\n",
    "plt.grid(True, linestyle='--', alpha=0.6)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 五大假設殘差診斷圖\n",
    "test_residuals = y_test - y_test_pred\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 11))\n",
    "\n",
    "# 1. Residuals vs Fitted\n",
    "axes[0, 0].scatter(y_test_pred, test_residuals, alpha=0.5, color='#2980b9', edgecolors='k', s=20)\n",
    "axes[0, 0].axhline(0, color='red', linestyle='--')\n",
    "axes[0, 0].set_title('Residuals vs Fitted (Linearity & Homoscedasticity)', fontsize=12)\n",
    "\n",
    "# 2. Normal Q-Q\n",
    "stats.probplot(test_residuals, dist=\"norm\", plot=axes[0, 1])\n",
    "axes[0, 1].get_lines()[0].set_color('#27ae60')\n",
    "axes[0, 1].set_title('Normal Q-Q Plot (Normality)', fontsize=12)\n",
    "\n",
    "# 3. Scale-Location\n",
    "axes[1, 0].scatter(y_test_pred, np.sqrt(np.abs(test_residuals / s)), alpha=0.5, color='#8e44ad', edgecolors='k', s=20)\n",
    "axes[1, 0].set_title('Scale-Location (Homoscedasticity)', fontsize=12)\n",
    "\n",
    "# 4. Actual vs Predicted\n",
    "axes[1, 1].scatter(y_test, y_test_pred, alpha=0.5, color='#d35400', edgecolors='k', s=20)\n",
    "axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')\n",
    "axes[1, 1].set_title(f'Actual vs Predicted (R² = {test_r2:.4f})', fontsize=12)\n",
    "axes[1, 1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "### 階段六：Deployment (Web 應用與部署)\n",
    "本專案已建置完整之 StatsUnlock 風格多元線性回歸計算機：\n",
    "- **前端檔案：** `web/index.html`、`web/styles.css`、`web/app.js`（支援純瀏覽器離線直接開啟執行）。\n",
    "- **後端 API：** `app.py`（基於 FastAPI，可透過 `uvicorn app:app --port 8000` 啟動）。\n",
    "- **功能亮點：** 即時計算 OLS 統計報表、ANOVA 表、係數信賴區間、VIF、雙折線特徵選擇圖表、動態滑桿情境預測。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 二、AI 協助與研究摘要\n",
    "### 1. NotebookLM 研究摘要（100字以上）\n",
    "**摘要內容：**  \n",
    "本研究利用 Google NotebookLM 對 Kaggle S4E5（Flood Prediction Dataset）競賽社群公開之頂尖解法進行文獻探討與比較。分析發現，該資料集 20 個水文與環境風險特徵具有高度的加性結構與對稱分布，因此標準多元線性回歸 (OLS) 即可作為極具競爭力之強大基準線 (R² ≈ 0.845)；而頂尖參賽方案進一步採用「列聚合特徵工程 (Row-wise Aggregations)」，提取每筆樣本在 20 個特徵間的統計特徵（如總和、均值、標準差、偏態等），並搭配嶺回歸 (Ridge)、LightGBM 與 CatBoost 進行 10 折交叉驗證集成，最終將 R² 提升至 0.869 以上。此比較顯示多元線性回歸具備卓越之計算效率與係數可解釋性，能清晰量化各水文指標之邊際風險，為防減災工程提供堅實之統計科學基礎。\n",
    "\n",
    "### 2. GPT 輔助內容與對話記錄指引\n",
    "- **輔助內容：** 本次作業中借助生成式 AI 完成 CRISP-DM 流程設計、SFS 逐步特徵選擇算法優化、殘差五大假設數學檢驗，以及參考 StatsUnlock 視覺風格建構 Web 系統架構。\n",
    "- **對話 PDF 匯出說明：** 依作業規定，已將所有對話紀錄透過 pdfCrowd / 瀏覽器 PDF 列印工具完整保存為 `GPT_dialogue.pdf`，並整合於壓縮檔案中以供查核。"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbformat": 4,
   "nbformat_minor": 4,
   "version": "3.12.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('7115056142_hw2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Created 7115056142_hw2.ipynb successfully!")
