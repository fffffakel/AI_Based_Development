# DA-2-35
1. Постройте корреляционную матрицу.
2. Найдите пары с |corr| > 0.8.
3. Выведите их.
4. Объясните, почему это проблема.


# Установка
1. Клонируйте репозиторий:
```
git clone https://github.com/fffffakel/AI_Based_Development/tree/main/DA-2-35
cd DA-2-35
```
2. Установите зависимости:
```
pip install -r requirements.txt
```

# Запуск
```
python analysis_corr.py
```

Чтобы использовать свои данные, измените path_to_dataset в  if \_\_name__ == "_\_main__": или передайте свой путь в функцию analyze_collinearity
