# ЛР3 - Воспроизведение эксперимента

Нужен Python 3.10+. 

Установка зависимостей и запуск тестов:

```bash
python -m venv .venv
source .venv/bin/activate # Linux
.venv/Scripts/activate # Windows
pip install -r requirements.txt
```

Чтобы воспроизвести полный эксперимент и построить графики:

```bash
python python lab3.py --runs 30 --first-seed 3000 --population 100 --generations 100
python plot_results.py
```

Результаты сохраняются в `results/`, графики — в `results/figures/`
