# Handoff — контекст для нового чата

Прочитай CLAUDE.md, docs/CASE.md, docs/FEATURES.md и src/config.py.

---

## Что уже сделано

### EDA (notebooks/01_eda.ipynb) — ЗАВЕРШЁН
- Загрузка 3 parquet файлов через Polars (business 3M tx / 25K карт, consumer 9.8M tx / 80K карт)
- Полный анализ: shape/dtypes/nulls/head/describe, сравнения, временные паттерны, мерчанты, качество данных
- Все графики через Plotly (интерактивные)
- Секция §3.5 — проверка новых гипотез (merchant_hhi, временные фичи, расширенный B2B-список)

### Feature Engineering (notebooks/02_features.ipynb) — ЗАВЕРШЁН
- Feature matrix построена: **105,000 карт × 64 фичи**, 0 nulls
- Сохранена в `data/processed/feature_matrix.parquet`
- Все группы A-J реализованы в `src/features/`:
  - `transactional.py` — Groups A, B, H, I
  - `mcc_based.py` — Group C
  - `temporal.py` — Group D (исправлен баг: Polars ISO weekday 1-7, config.py обновлён)
  - `recurring.py` — Group E
  - `geo_channel.py` — Groups F, G
  - `graph_features.py` — Group J (cosine similarity + merchant overlap)
  - `build_features.py` — оркестратор

### Модель (notebooks/03_model.ipynb) — КОД НАПИСАН, ещё не запущен
- `src/models/baseline.py` — Logistic Regression pipeline
- `src/models/pu_bagging.py` — `PUBaggingClassifier` (LightGBM, N=10 итераций, OOB-скоры)
- `src/models/catboost_model.py` — `PUBaggingCatBoost` challenger
- `src/models/train.py` — `load_feature_matrix()`, `prepare_splits()`, `save_model()`
- `src/evaluation/metrics.py` — ROC-AUC, PR-AUC, F1, Precision@K
- `src/evaluation/plots.py` — все стандартные графики
- `src/evaluation/injection_test.py` — Synthetic Injection Test

---

## Ключевые находки (подтверждены на реальной feature matrix)

### Типы данных — ВАЖНО
- `mcc` = **String** — все `is_in` через `[str(c) for c in B2B_MCC_CODES]`
- `transaction_amount_kzt` = **Int64** — кастить к Float64 перед агрегациями
- `Is_recurring` → нормализовать в `is_recurring` при загрузке
- **Polars `dt.weekday()` = ISO weekday (1=Mon, 7=Sun)** — `config.py` исправлен: `BUSINESS_DAYS={1,2,3,4,5}`, `WEEKEND_DAYS={6,7}`

### Реальные медианы из feature matrix (105K карт)
| Фича | Business | Consumer | Pearson r с label |
|------|----------|----------|--------------------|
| `b2b_spend_share` | **81.1%** | **0.0%** | ~+0.75 |
| `night_recurring_share` | 13.4% | **0.0%** | ~+0.55 |
| `weekend_share` | **12.4%** | **35.0%** | ~-0.70 ✓ (исправлено) |
| `median_ticket_kzt` | 84,559 ₸ | 9,674 ₸ | — |
| `night_share` | 15.0% | 5.4% | ~+0.50 |
| `recurring_share` | 13.4% | **0.0%** | ~+0.80 |
| `merchant_hhi` | **0.224** | **0.102** | — |
| `business_hours_share` | **60.2%** | **33.7%** | — |
| `lunch_dip_ratio` | **0.727** | **1.0** | — |

### Сюрпризы из корреляционного анализа (не было в EDA)
- **`foreign_tx_share` — Pearson r~+0.90** — самый коррелированный признак. Бизнес тратит у иностранных мерчантов (Google Ads, AWS, SaaS).
- **`business_merchant_overlap` — r~+0.85** — граф-фича работает отлично (#2 по корреляции).
- **`evening_share` — r~-0.90** — сильнейший отрицательный. Потребители активны 18-23ч, бизнес — нет.
- **`dow_entropy` — r~-0.85** — бизнес работает по регулярному недельному расписанию.
- **`recurring_amount_share` — r~+0.80** — доля денег на SaaS-подписках.

---

## Следующий шаг — Запустить 03_model.ipynb

Ноутбук написан, нужно запустить. Ожидаемое время: ~15-20 мин (10 итераций PU-bagging).

После запуска 03_model.ipynb — обнови этот HANDOFF с реальными метриками модели, затем переходи к **04_segmentation.ipynb**.

### Что ожидать от модели
- PR-AUC > 0.85 (при сильных фичах типа `foreign_tx_share` и `business_merchant_overlap`)
- Injection Test Recall@top5% > 0.7 (желательно > 0.8)
- Если метрики ниже — проверь SHAP: топ-10 SHAP должны совпадать с топом корреляций

### После 03_model.ipynb — задачи
1. **04_segmentation.ipynb** — KMeans (k=5) на найденных hidden entrepreneurs, профили сегментов
   - `src/segmentation/cluster.py` — пустой, нужно написать
2. **Optuna тюнинг** — если время позволяет, добавить в `pu_bagging.py` перебор гиперпараметров
3. **Финальный пайплайн** — `notebooks/FINAL_pipeline.ipynb` воспроизводимый от сырых данных до результатов

---

## Структура проекта (актуальная)

```
src/
├── config.py              ← единственный источник констант
├── features/
│   ├── build_features.py  ← build_feature_matrix() — РАБОТАЕТ
│   ├── transactional.py, mcc_based.py, temporal.py
│   ├── recurring.py, geo_channel.py, graph_features.py
├── models/
│   ├── baseline.py, pu_bagging.py, catboost_model.py, train.py  ← написаны
├── evaluation/
│   ├── metrics.py, plots.py, injection_test.py  ← написаны
└── segmentation/
    └── cluster.py  ← ПУСТОЙ, нужно написать

data/processed/
├── feature_matrix.parquet  ← ГОТОВ (105K × 64)
└── consumer_scored.parquet  ← создаётся после 03_model.ipynb

models/
└── pu_bagging_lgbm.pkl  ← создаётся после 03_model.ipynb
```
