# Notebooks

这个目录放 Jupyter 版本的交互式学习材料。

推荐从项目根目录启动：

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m ipykernel install --user --name learn-ai --display-name "Python (learn-ai)"
uv run jupyter lab notebooks
```

如果你更习惯经典 Notebook 界面：

```bash
uv run jupyter notebook notebooks
```

如果 notebook 里出现 `ModuleNotFoundError: No module named 'matplotlib'`，通常是当前页面选错了 kernel。请在 Jupyter 右上角或菜单里切换到 `Python (learn-ai)`，然后重启 kernel 并重新运行 cell。

学习建议：

1. 先运行每个 cell，不急着改代码。
2. 看懂输出里的 shape、mask、概率分布。
3. 修改 notebook 最后的练习参数，再重新运行相关 cell。
4. 对照 `examples/` 里的脚本版本，理解 notebook 和脚本如何对应。
