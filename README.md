# Mastercard Data Quest 2026 — Hidden Entrepreneur Detection

**Хакатон:** AIESEC × Mastercard Data Quest
**Период:** 23 мая — 30 мая 2026
**Город:** Алматы

---

## Задача

Разработать ML-алгоритм, который по транзакционному поведению обнаруживает скрытую коммерческую активность среди клиентов-физлиц (consumer cards).

**Тип задачи:** Positive-Unlabeled (PU) Learning
- Dataset X (business, 25K карт, ~3M tx) — известный позитив
- Dataset Y (consumer, 80K карт, ~10M tx) — смесь обычных потребителей и скрытых бизнесов

**Финальная метрика (Q2 организаторов):** ROC-AUC

**Сдаваемые материалы:**
1. `notebooks/FINAL.ipynb` — единый воспроизводимый ноутбук (требование Q1)
2. `submission_combined.csv` — `card_number, score` для всех 80K consumer карт (рекомендуемая версия)
3. `submission.csv` — альтернативная версия (PU-Bagging only)
4. PowerPoint презентация (отдельно)

---

## Результаты

| Метрика | Значение | Где |
|---------|----------|-----|
| **ROC-AUC** (5-fold CV) | **1.0000 ± 3e-8** | `reports/diagnostics/cv_summary.csv` |
| ROC-AUC (holdout) | 1.0000 | `reports/diagnostics/holdout_metrics.csv` |
| Confusion Matrix | TP=5000, FP=1, FN=0, TN=15999 | `reports/diagnostics/confusion_matrix.png` |
| Ablation: 13 фичей из 63 | ROC-AUC = 0.9999 | `reports/diagnostics/ablation_results.csv` |
| Топ-кандидаты (score > 0.001) | 165 карт | `data/processed/consumer_scored.parquet` |

---

## Быстрый старт

```bash
# 1. Клонировать репо
git clone <repo-url>
cd mdq-hackathon

# 2. Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Положить сырые данные в data/raw/
#    business_cards_MDQ.parquet
#    consumer_cards_MDQ.parquet
#    merchants_reference.parquet

# 5. Запустить финальный ноутбук
jupyter lab notebooks/FINAL.ipynb
```

При первом запуске ноутбук пересчитывает feature matrix и обучает модели (~3 мин). Последующие запуски используют кэш из `data/processed/` и `models/` (~30 сек).

---

## Структура проекта

```
mdq-hackathon/
├── notebooks/
│   ├── FINAL.ipynb              ⭐ ОСНОВНОЙ DELIVERABLE (54 ячейки, end-to-end)
│   ├── 01_eda.ipynb             — exploratory analysis (legacy)
│   ├── 02_features.ipynb        — feature engineering experiments (legacy)
│   ├── 03_model.ipynb           — model training notes (legacy)
│   └── 04_segmentation.ipynb    — segmentation analysis (legacy)
├── submission.csv               ⭐ PU-Bagging only (80,000 rows)
├── submission_combined.csv      ⭐ PU × Anomaly Boost (рекомендуемая версия)
├── src/                         — весь переиспользуемый код
│   ├── config.py                — пути, MCC-коды, ROI assumptions
│   ├── features/                — feature engineering (6 модулей по группам A-J)
│   ├── models/                  — baseline + PU-Bagging
│   ├── evaluation/              — CV, diagnostics, SHAP
│   └── segmentation/            — KMeans + профили
├── scripts/                     — pipeline-скрипты (запускают этапы вне ноутбука)
│   ├── retrain_after_fix.py     — переобучение модели
│   ├── phase2_validation.py     — 5-fold CV + Confusion Matrix
│   ├── phase2f_shap.py          — SHAP explainability
│   ├── phase1_5_anomaly_boost.py — Anomaly boost эксперимент
│   ├── phase6a_archetypes.py    — 3 архетипа с реальными мерчантами
│   ├── phase6b_pca_viz.py       — PCA 2D visualization
│   ├── phase6c_segments_radar.py — Segment radar charts
│   ├── phase6d_ablation.py      — Ablation test
│   └── build_final_notebook.py  — генератор FINAL.ipynb
├── reports/diagnostics/         — все артефакты валидации (CV, CM, SHAP, PCA, segments, ablation, archetypes)
├── data/
│   ├── raw/                     — исходные parquet (не в git)
│   └── processed/               — feature_matrix.parquet, consumer_scored.parquet
├── models/                      — baseline_logreg.pkl, pu_bagging_lgbm.pkl
└── docs/                        — документация
    ├── CASE.md                  — описание кейса + Q&A organizers
    ├── SOLUTION.md              — архитектура ML-решения
    ├── FEATURES.md              — описание всех фичей с бизнес-обоснованием
    ├── HANDOFF.md               — текущий статус (живой документ)
    ├── STACK.md                 — технический стек
    ├── CRITERIA.md              — критерии оценки жюри
    └── GIT_WORKFLOW.md          — правила работы с git
```

---

## Архитектура решения

```
Raw Data (13M транзакций)
        ↓
  Feature Engineering (Polars, 63 фичи на карту)
        ↓
  ┌─────────────────────────────────┐
  │   PU-Bagging Classifier         │ ← Основная модель (LightGBM, 10 итераций)
  │   business=1, consumer=0        │
  └─────────────────────────────────┘
        ↓
  Business-Score для каждой consumer-карты
        ↓
  ┌─────────────────────────────────┐
  │   Anomaly Boost (Phase 1.5)     │ ← IsolationForest на consumer
  │   combined = √(rank_PU × rank_anomaly) │
  └─────────────────────────────────┘
        ↓
  Submission (card_number, score) для 80K карт
        ↓
  Segmentation (KMeans k=5) → product recommendations
```

---

## Документация

| Файл | Содержание |
|------|-----------|
| [docs/CASE.md](docs/CASE.md) | Описание кейса + раздел Q&A organizers (timezone, метрика, формат) |
| [docs/SOLUTION.md](docs/SOLUTION.md) | Архитектура ML-решения |
| [docs/FEATURES.md](docs/FEATURES.md) | Все фичи с бизнес-обоснованием + SHAP таблица |
| [docs/HANDOFF.md](docs/HANDOFF.md) | Текущий статус проекта (живой документ) |
| [docs/STACK.md](docs/STACK.md) | Технический стек |
| [docs/CRITERIA.md](docs/CRITERIA.md) | Критерии оценки жюри |
| [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) | Git правила |
