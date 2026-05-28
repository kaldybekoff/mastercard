# Технический стек — MDQ 2026

## Среда разработки

| Инструмент | Версия | Роль |
|-----------|--------|------|
| Python | 3.13 | Основной язык |
| venv | стандартный | Управление средой (`.venv` в корне репо) |
| VSCode | latest | IDE (Jupyter extension) |
| Jupyter Notebook | — | Формат финального решения (требование Q1/Q7 организаторов) |

**Запуск среды (Windows):**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Стек по слоям

### Data Processing

| Библиотека | Минимальная версия | Зачем |
|-----------|--------|-------|
| **Polars** | ≥1.12 | Основная: агрегации на 13M строк (5-10× быстрее Pandas) |
| **Pandas** | ≥2.2 | Только для передачи в sklearn / LightGBM / SHAP |
| **PyArrow** | ≥18.0 | Backend для чтения parquet |
| **NumPy** | ≥1.26 | Векторные операции |
| **SciPy** | ≥1.14 | `rankdata`, `spearmanr` для Anomaly Boost |

**Правило команды:** весь feature engineering — на Polars. В Pandas конвертируем только перед `model.fit()`.

```python
# Правильно
df = pl.read_parquet("data/raw/business_cards_MDQ.parquet")
features = df.group_by("card_number").agg([...])

# Передача в модель
X = features.to_pandas()
```

### Machine Learning

| Библиотека | Роль |
|-----------|------|
| **LightGBM** | Основная модель (PU-Bagging classifier, 10 итераций) |
| **scikit-learn** | LogReg baseline, KMeans, IsolationForest (anomaly boost), StandardScaler, PCA, метрики |
| **joblib** | Сохранение моделей в `.pkl` |

### Explainability

| Библиотека | Зачем |
|-----------|-------|
| **SHAP** | TreeExplainer для LightGBM — summary_plot + waterfall_plot |

### Metrics & Visualization

| Библиотека | Зачем |
|-----------|-------|
| **scikit-learn** | confusion_matrix, ROC-AUC, PR-AUC, F1, Precision, Recall |
| **Matplotlib** | Все графики проекта |
| **tabulate** | Markdown-таблицы в отчётах |

### Сохранение

| Формат | Когда |
|--------|-------|
| `.pkl` (joblib) | LightGBM, sklearn-объекты |
| `.parquet` | Feature matrix + scored consumer |
| `.csv` | Submission, диагностика, ablation |
| `.png` | Визуализации (PCA, SHAP, confusion matrix, segment radars) |

---

## Что НЕ используем и почему

| Инструмент | Почему нет |
|-----------|-----------|
| XGBoost | Дублирует LightGBM, хуже с категориалами |
| CatBoost | Заявляли как challenger, но 5-fold CV LightGBM = 1.0000 — challenger не нужен |
| TensorFlow / PyTorch | Табличная задача, DL не даёт выигрыша |
| Streamlit / Dash | Дашборд не сдаётся (требование Q1 — один Jupyter файл). Threshold не нужен в submission |
| Optuna | CV ROC-AUC уже = 1.0, тюнить нечего |
| PySpark | Данные помещаются в RAM (13M строк) |

---

## Структура `src/`

```
src/
├── __init__.py
├── config.py                — пути, MCC-коды, ROI assumptions, RANDOM_STATE
├── features/                — feature engineering по группам
│   ├── transactional.py     — Group A/B/H/I: объём, diversity, velocity, card metadata
│   ├── mcc_based.py         — Group C: B2B / Consumer / Mixed / Rental MCC features
│   ├── temporal.py          — Group D: business_hours, weekday, evening, entropy
│   ├── recurring.py         — Group E: recurring patterns
│   ├── geo_channel.py       — Group F/G: online/offline, foreign tx (FIXED: country="Kazakhstan", не "KZ")
│   ├── graph_features.py    — Group J: bipartite merchant overlap + cosine similarity
│   └── build_features.py    — главная функция: build_feature_matrix()
├── models/
│   ├── baseline.py          — Logistic Regression
│   ├── pu_bagging.py        — PU-Bagging с LightGBM (главная модель)
│   └── train.py             — prepare_splits() + save/load helpers
├── evaluation/
│   ├── cv.py                — run_5fold_cv() + summarize_cv()
│   ├── diagnostics.py       — score distribution, top-N inspection, confusion matrix
│   ├── shap_analysis.py     — TreeExplainer wrapper + global/local plots
│   ├── metrics.py           — utility (legacy)
│   └── plots.py             — utility plots (legacy)
└── segmentation/
    └── cluster.py           — KMeans + segment profiling
```

---

## Структура `scripts/`

Оркестраторы, запускающие компоненты из `src/`. Все используют `.venv\Scripts\python.exe`.

| Скрипт | Назначение |
|--------|------------|
| `retrain_after_fix.py` | Полный pipeline: feature build → split → train baseline + PU → score 80K |
| `phase2_validation.py` | 5-fold CV + Confusion Matrix + Score Distribution + Top-50 |
| `phase2f_shap.py` | SHAP global + 3 local (high/mid/low score) |
| `phase1_5_anomaly_boost.py` | IsolationForest + combined rank score |
| `phase6a_archetypes.py` | 3 архетипа hidden entrepreneurs с реальными мерчантами |
| `phase6b_pca_viz.py` | 2D PCA визуализация |
| `phase6c_segments_radar.py` | KMeans k=5 + radar charts |
| `phase6d_ablation.py` | Ablation test (7 экспериментов с разными feature subsets) |
| `build_final_notebook.py` | Генератор `notebooks/FINAL.ipynb` через nbformat |
