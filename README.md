# Разработка мультиязычной модели информационного поиска на основе ColBERT для русского языка

Репозиторий содержит код и выполненные ноутбуки к дипломной работе.

## `code/`

Скрипты пайплайна обучения RuColBERT и оценки:

| Файл | Назначение |
|------|------------|
| `config.py` | Конфигурации модели, данных и обучения |
| `modeling.py` | Модель ColBERT, токенизатор |
| `data.py` | Загрузка и подготовка данных, негативы, даталоадеры |
| `train.py` | Обучение (фазы с лёгкими и hard-негативами) |
| `index_and_retrieve.py` | Индексация и ретривал |
| `evaluate.py` | Локальная оценка качества |
| `evaluate_mteb.py` | Оценка на  ruMTEB бенчмарказ |
| `benchmark_dense_baselines.py` | Сравнение с базовым bi-encoder |

## `diploma_finished_notebooks/`

Воспроизводимые **Jupyter-ноутбуки** экспериментов:

- **Обучение:** `training-phase1.ipynb`, `training-phase2-300k.ipynb`, ранние этапы и подготовка данных — `1st_stages_diploma.ipynb`
- **Майнинг негативов:** `mining_hard_negatives.ipynb`
- **Индексация:** `index-and-retrieve.ipynb`
- **Оценка:** `evaluate-mteb.ipynb`, `evaluate-mteb-dense-with-bm25.ipynb`, `evaluate_robustness_dense.ipynb`, `eval_robustness.ipynb`, `eval-robustness-zeroshot.ipynb`
- **Абляции экспы:** `ablation-no-eng-phase1.ipynb`, `ablation-no-eng-phase2.ipynb`
- **Замер системных метрик:** `benchmark_latency.ipynb`

Ноутбуки привязаны к окружению с установленными зависимостями из экспериментов (PyTorch, Hugging Face, datasets, при необходимости MTEB и др.). Выполнялись в сервисах Google Colab и Kaggle.
