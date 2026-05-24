# Технический стек — MDQ 2026

## Среда разработки

| Инструмент | Версия | Роль |
|-----------|--------|------|
| Python | 3.11 | Основной язык |
| Anaconda | latest | Управление средой |
| VSCode | latest | IDE (Jupyter extension) |
| Jupyter Notebook | — | Формат финального решения (требование хакатона) |

**Запуск среды:**
```bash
conda create -n mdq python=3.11
conda activate mdq
pip install -r requirements.txt
```

---

## Стек по слоям

### Data Processing

| Библиотека | Версия | Зачем |
|-----------|--------|-------|
| **Polars** | ≥1.12 | Основная: агрегации на 13M строк. В 5-10x быстрее Pandas |
| **Pandas** | ≥2.2 | Только для передачи в sklearn/LightGBM |
| **PyArrow** | ≥18.0 | Backend для чтения parquet |
| **NumPy** | <2.0 | Векторные операции |

**Правило команды:** весь feature engineering — на Polars. В Pandas конвертируем только перед `model.fit()`.

```python
# Правильно
df = pl.scan_parquet("data/raw/business_cards_MDQ.parquet")
features = df.group_by("card_number").agg([...]).collect(streaming=True)

# Передача в модель
X = features.to_pandas()
```

### Machine Learning

| Библиотека | Роль |
|-----------|------|
| **LightGBM** | Основная модель (PU-bagging classifier) |
| **CatBoost** | Challenger-модель для сравнения |
| **scikit-learn** | LogReg baseline, KMeans, Isolation Forest, метрики |
| **Optuna** | Гиперпараметр-тюнинг (Bayesian, 100+ trials) |

### Explainability

| Библиотека | Зачем |
|-----------|-------|
| **SHAP** | TreeExplainer для LightGBM — summary_plot + waterfall_plot |

### Metrics & Visualization

| Библиотека | Зачем |
|-----------|-------|
| **scikit-learn** | confusion_matrix, PR-AUC, ROC-AUC, F1, classification_report |
| **Matplotlib** | Базовые графики |
| **Seaborn** | Heatmaps (confusion matrix), distributions |
| **Plotly** | Интерактивные графики в Streamlit |

### Dashboard

| Библиотека | Зачем |
|-----------|-------|
| **Streamlit** | Основной инструмент для дашборда |
| **Plotly** | Интерактивные чарты внутри Streamlit |

**Почему Streamlit, не Power BI:**
- Устанавливается за 1 команду, нет лицензии
- Python-native — весь ML-код доступен прямо в приложении
- Можно интегрировать SHAP, ROI-калькулятор, threshold-слайдер за часы
- Показывается в браузере, не нужно ничего открывать отдельно

### Сохранение моделей

| Формат | Когда |
|--------|-------|
| `.pkl` (joblib) | LightGBM, sklearn-объекты |
| `.cbm` | CatBoost native формат |
| `.parquet` | Feature matrix (processed data) |

---

## requirements.txt

```
# Data processing
polars>=1.12.0
pandas>=2.2.0
pyarrow>=18.0.0
numpy>=1.26.0,<2.0.0

# Machine learning
scikit-learn>=1.5.0
lightgbm>=4.5.0
catboost>=1.2.7
optuna>=4.0.0

# Explainability
shap>=0.46.0

# Visualization
matplotlib>=3.9.0
seaborn>=0.13.0
plotly>=5.24.0

# Dashboard
streamlit>=1.40.0

# Notebook
jupyterlab>=4.3.0
ipykernel>=6.29.0

# Utils
tqdm>=4.66.0
pyyaml>=6.0.0
joblib>=1.4.0
```

---

## Что НЕ используем и почему

| Инструмент | Почему нет |
|-----------|-----------|
| XGBoost | Дублирует LightGBM, хуже с категориалами |
| TensorFlow / PyTorch | Табличная задача, DL не даст выигрыша за 7 дней |
| Power BI / Tableau | Лицензия, внешняя экосистема, сложно деплоить |
| Plotly Dash | Overkill по сложности vs Streamlit |
| PySpark | Нет кластера, данные помещаются в RAM |

---

## Структура `src/`

```
src/
├── __init__.py
├── config.py              # Пути, константы, списки B2B MCC кодов
├── features/
│   ├── __init__.py
│   ├── transactional.py   # Группы A, B, H — объём, diversity, velocity
│   ├── mcc_based.py       # Группа C — MCC классификация и фичи
│   ├── temporal.py        # Группа D — временные паттерны
│   ├── geo_channel.py     # Группы F, G — канал, география
│   ├── recurring.py       # Группа E — регулярность
│   ├── graph_features.py  # Группа J — merchant graph (бонус)
│   └── build_features.py  # Главная функция: build_feature_matrix()
├── models/
│   ├── __init__.py
│   ├── baseline.py        # Logistic Regression
│   ├── pu_bagging.py      # PU-bagging с LightGBM
│   ├── catboost_model.py  # CatBoost challenger
│   └── train.py           # Единый интерфейс train/predict/save
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py         # confusion_matrix, PR-AUC, ROC-AUC, F1, Precision@K
│   ├── plots.py           # Стандартные графики для notebook и презентации
│   └── injection_test.py  # Synthetic Injection Test
└── segmentation/
    ├── __init__.py
    └── cluster.py         # KMeans + профилирование сегментов
```
