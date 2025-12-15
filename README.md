# COMMRivew-BERTopic
本项目是2025全球传播学研究综述数据处理程序，代码脚本主要用于执行自动化 BERTopic 模型训练和参数选择。通过网格搜索（Grid Search）的方式，测试多组超参数组合，并基于主题一致性评估每组参数的效果，可以找到最优的模型配置。

## 🛠️ 环境与依赖

项目基于 Python 3.12 开发，主要依赖以下库：

- `bertopic`
- `sentence-transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `umap-learn`
- `nltk`
- `gensim`
- `openpyxl` 

### 1. 安装依赖

你可以通过 pip 一键安装所有必要的库：

```bash
pip install bertopic sentence-transformers pandas numpy scikit-learn umap-learn nltk gensim openpyxl
```

### 2. 下载 NLTK 数据

脚本会自动检测并下载所需的 NLTK 数据包（`punkt`, `averaged_perceptron_tagger`, `wordnet`, `stopwords`）。首次运行时，请确保你的网络连接正常。

## 🚀 如何运行

### 1. 准备输入数据

将你的数据保存在一个 Excel 文件中。该文件应至少包含两列，脚本默认读取 `Article Title` 和 `Abstract` 列。你可以在脚本中修改 `FILE_PATH` 和列名。

- **`FILE_PATH`**: 将此变量设置为你的数据文件路径。

```python
# 脚本中的参数
FILE_PATH = 'communication_2578.xlsx'
```

### 2. 配置参数

在运行脚本之前，你可以根据需求调整以下参数：

- **`EMBEDDING_MODEL_NAME`**: 词嵌入模型，默认为 `all-mpnet-base-v2`。
- **`PARAM_TARGET_NUM_TOPICS`**: 网格搜索的目标主题数列表。
- **`PARAM_UMAP_N_NEIGHBORS`**: UMAP 的 `n_neighbors` 参数列表。
- **`PARAM_MIN_TOPIC_SIZE`**: 最小主题大小参数列表。

```python
# 网格搜索参数
PARAM_TARGET_NUM_TOPICS = [12, 14, 16, 18, 20, 22, 24]
PARAM_UMAP_N_NEIGHBORS = [5, 10, 15]
PARAM_MIN_TOPIC_SIZE = [5, 10]
```

### 3. (可选) 自定义预处理

- **自定义停用词**: 在项目根目录下创建 `custom_stopwords.txt` 文件，每行写入一个不希望出现在主题模型中的词。
- **自定义短语**: 创建 `custom_phrases.txt` 文件，每行写入一个希望被当作单个词处理的短语（例如 `machine learning`）。脚本会自动将其转换为 `machine_learning`。

### 4. 运行脚本

在终端中执行 Python 脚本：

```bash
python your_script_name.py
```

脚本将开始执行，并在控制台输出每个参数组合的运行进度和结果。

## 📊 输出结果

所有输出文件都将保存在由 `OUTPUT_DIR` 参数指定的目录中（默认为 `communication_grid_search_results`）。

### 1. 单次运行结果

对于每一组参数组合（例如 `T=12, N=15, S=10`），会生成一组文件，文件名前缀格式为 `t{主题数}_n{邻近点数}_s{主题大小}`：

- `t12_n15_s10_model/`: 保存的 BERTopic 模型文件夹。
- `t12_n15_s10_visualization.html`: 可交互的主题可视化图。
- `t12_n15_s10_detail.xlsx`: 原始数据及每篇文档对应的主题ID和关键词。
- `t12_n15_s10_summary.xlsx`: 生成的主题列表、关键词和文档数量统计。

### 2. 参数网格搜索

BERTopic 模型参数调优算法。

## 🌟 具体流程

- **自动化网格搜索**: 自动遍历预设的多组参数（目标主题数、UMAP邻近点数、最小主题大小），无需手动反复运行。
- **两阶段优化策略**: 采用先合并主题（`reduce_topics`）再减少离群点（`reduce_outliers`）的策略，以获得更稳定和可控的主题模型。
- **全面的结果输出**: 为每一组参数组合，脚本会保存：
    - 训练好的 BERTopic 模型。
    - 可交互的主题可视化 HTML 文件。
    - 包含每篇文档主题分配的详细 Excel 文件。
    - 主题信息（关键词、文档数）的摘要 Excel 文件。
- **最终结果汇总**: 网格搜索结束后，会生成一个总的 Excel 文件，按主题一致性得分对所有参数组合进行排序，方便快速找到最佳配置。
- **可定制的文本预处理**: 支持通过外部文件（`custom_stopwords.txt`, `custom_phrases.txt`）自定义停用词和需要保留的短语。

- **注意**：`grid_search_coherence_summary.xlsx`: 这是最重要的汇总文件。它列出了所有参数组合及其对应的主题一致性得分（`Coherence_CV`）、初始/最终主题数等信息，并按一致性得分从高到低排序。你可以直接查看此文件的第一行来确定最佳参数组合。

## 📝 其他注意事项

- **计算资源**: 嵌入向量的计算和网格搜索过程可能会非常耗时，尤其是在数据集较大或参数组合较多时。建议在有足够计算资源的机器上运行。
- **嵌入模型下载**: 首次运行会从 Hugging Face Hub 下载 `all-mpnet-base-v2` 模型（约 400MB），请确保网络通畅。模型下载后会缓存到本地，后续运行无需再次下载。
- **内存占用**: 处理大型数据集时，请关注内存使用情况，避免因内存不足导致程序中断。
