# 让 `python -m unittest discover -s tests -t .` 能工作。
# 指定 -t（top-level dir）为仓库根时，unittest 要求起始目录本身可导入，
# 否则报 "Start directory is not importable"。
