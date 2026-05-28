# Git Workflow — MDQ 2026

## Правила

1. `main` — всегда рабочая ветка. FINAL.ipynb запускается без ошибок.
2. Любая работа → отдельная ветка от `main`
3. Имя ветки: `feature/имя-задача` (например `feature/ali-temporal-features`)
4. Merge в `main` минимум раз в день
5. Прямые коммиты в `main` — только README, docs, мелкие правки

## Ноутбуки и конфликты

**Главное правило:** каждый ноутбук принадлежит одному человеку. Никто не лезет в чужой ноутбук.

Когда код в ноутбуке **работает** → переносим в `src/` → ноутбук остаётся только с импортами и вызовами.

## Cheatsheet

```bash
# Начало работы
git checkout main && git pull
git checkout -b feature/моя-задача

# Работа...
git add .
git commit -m "feat: описание что сделано"

# Перед push
git checkout main && git pull
git checkout feature/моя-задача
git rebase main
git push -u origin feature/моя-задача
# → Pull Request → Merge
```

## .gitignore (критично)

Данные, модели и кэш **никогда не коммитим:**
```
data/raw/*
data/interim/*
data/processed/*
models/*.pkl
models/*.cbm
*.parquet
.ipynb_checkpoints/
__pycache__/
.venv/
```
