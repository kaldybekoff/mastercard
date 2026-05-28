# Архитектура решения — Hidden Entrepreneur Detection

## Постановка задачи

**Тип задачи:** Positive-Unlabeled (PU) Learning

Скрытый предприниматель ≠ копия business cardholder. Business card используется исключительно для бизнес-расходов (легализованный бизнес). Скрытый предприниматель смешивает личные и бизнес-траты на одной карте.

> Мы ищем consumer-карты, чьё поведение **сдвинуто в сторону business**, но не идентично ему.

---

## Общая архитектура пайплайна

```
Raw Data (13M транзакций)
        ↓
  Feature Engineering
  (транзакции → ~80 фичей на карту → 105K строк)
        ↓
  ┌─────────────────────────────────┐
  │    PU-Bagging Classifier        │ ← Основная модель (LightGBM)
  │    business=1, consumer=0       │
  └─────────────────────────────────┘
        ↓
  Business-Score (0→1) для каждой consumer-карты
        ↓
  Threshold Selection (бизнес-решение)
        ↓
  Hidden Entrepreneurs List (~5K карт)
        ↓
  Segmentation (KMeans)
        ↓
  Segment Profiles + Product Recommendations
```

---

## Слой 1: PU-Bagging Classifier (основная модель)

### Алгоритм

```
Positives (P):  25K business cards  → label = 1
Unlabeled (U):  80K consumer cards  → смесь, без label

Repeat N=10 итераций:
  1. Сэмплируем U_i = 25K случайных consumer → label = 0
  2. Обучаем LightGBM на (P ∪ U_i)
  3. Предсказываем на (Consumer \ U_i) → score_i

Final_score(card) = mean(score_i) по out-of-bag итерациям
```

### Почему PU-bagging, а не наивный бинарный классификатор

Наивный подход (business=1, consumer=0) принимает **всех consumer за чистых потребителей**. Это неверно — среди 80K consumer есть скрытые предприниматели. PU-bagging снижает этот bias через усреднение по множеству случайных подвыборок.

### Основная модель: LightGBM

**Обоснование выбора:**
- Нативная поддержка категориальных фичей (MCC, channel, bank_name) без OHE
- Быстрее CatBoost в 2-3x на тренировке → больше Optuna trials
- `scale_pos_weight` для имбаланса классов
- Стандарт для табличного ML → жюри это понимает

**Challenger-модель: CatBoost** — для сравнения и подтверждения результатов.

**Baseline: Logistic Regression** — обязательно, для методологической чистоты и интерпретируемости.

---

## Слой 2: Validation Strategy

### Holdout split

```
Business cards (25K):
  → Train: 20K (80%)
  → Holdout: 5K (20%) ← модель НИКОГДА не видит при обучении

Consumer cards (80K):
  → Все используются как unlabeled (inference)
```

Важно: split на уровне **карт**, не транзакций. Иначе data leakage.

### Synthetic Injection Test (уникальная валидация)

Проблема PU-learning: в consumer нет ground truth → нельзя измерить recall напрямую.

**Решение:**
```
1. Берём 1000 business-карт из holdout
2. Убираем их метку, помещаем в consumer pool
3. Запускаем полный pipeline
4. Проверяем: какая доля из 1000 попала в top-5% scoring карт?
→ Это прямая оценка recall на скрытых предпринимателях
```

---

## Метрики

| Метрика | Зачем |
|---------|-------|
| **ROC-AUC** | **ОСНОВНАЯ** — подтверждено Q2 организаторами: финальная оценка вручную по ROC-AUC |
| **Confusion Matrix** | Обязательное требование критерия №2 |
| **PR-AUC** | Дополнительная — диагностическая для проверки на дисбаланс |
| **Precision@K** | Бизнес-метрика: из top-K карт, какая доля = реально бизнес |
| **F1** | Дополнительная |

**Почему ROC-AUC:**
По Q2 от партнёров Mastercard, финальная оценка submissions проводится вручную организаторами по метрике ROC-AUC. Они сравнивают наши `(card_number, score)` со скрытыми истинными метками в Dataset Y и считают ROC-AUC. Это **rank-based** метрика — она измеряет качество ранжирования, не значения скоров.

---

## Слой 3: Segmentation

После выделения hidden entrepreneurs (порог = бизнес-решение, управляется слайдером в дашборде):

```
KMeans (k=4-6) на feature vectors найденных карт
→ Segment profiles
→ Product recommendations per segment
```

### Ожидаемые сегменты

| Сегмент | Признаки | Продукт |
|---------|----------|---------|
| E-commerce sellers | Высокий online_share, маркетплейс-MCC | POS-онлайн-эквайринг |
| Offline retail | Высокий offline_share, оптовые MCC | Торговый эквайринг + кредит оборотки |
| Service providers | Google Ads / Meta Ads spend, низкие закупки | Бизнес-карта + cashback на рекламу |
| Restauranteurs | Food wholesale MCC, утренняя активность | Ресторанный эквайринг |
| Digital / IT | SaaS recurring, иностранные B2B | Мультивалютная бизнес-карта |

---

## Explainability

**SHAP TreeExplainer** на финальной LightGBM-модели:

- **Global:** summary_plot — топ фичей по важности для всей модели
- **Local:** waterfall_plot — почему конкретная карта получила высокий score

Пример вывода на питче:
> "Карта X получила score 0.94. Главные факторы: (1) 78% транзакций в рабочие часы vs медиана consumer 32%, (2) 7 уникальных B2B-MCC кодов vs медиана consumer 0, (3) низкая концентрация мерчантов — 320 уникальных vs медиана consumer 45."

---

## ROI-расчёт (для финального слайда)

| Параметр | Значение | Источник |
|----------|----------|----------|
| Hidden entrepreneurs найдено | ~4 800 | 6% от 80K consumer (assumption) |
| Conversion rate | 15% | Industry benchmark МСБ-кампаний |
| Конвертированных клиентов | 720 | 4800 × 15% |
| LTV (1 год) | 200K₸ | Эквайринг + обслуживание + кредит |
| **Annual upside** | **144M₸** | 720 × 200K₸ |

> Все цифры — assumptions, явно помечены в презентации.

---

## Ограничения решения

1. **Нет income-side данных** — входящие P2P-переводы (главный сигнал) недоступны. Снижает recall на сегменте услуг.
2. **Синтетика** — модель может выучить артефакты генерации. Митигация: проверяем SHAP, убеждаемся что топ-фичи бизнес-осмысленны.
3. **PU-learning → ranking, не абсолютная правда** — threshold выбирается бизнесом, не моделью.
4. **Сегменты unsupervised** — нуждаются в валидации банковским экспертом.
