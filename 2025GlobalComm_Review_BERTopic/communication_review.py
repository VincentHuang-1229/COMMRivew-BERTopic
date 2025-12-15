
import os
import time
import warnings
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from collections import Counter

warnings.filterwarnings("ignore")

# ----------------------------
# 参数区
# ----------------------------
FILE_PATH = 'communication_2578.xlsx'
OUTPUT_DIR = 'communication_grid_search_results'  # 用于保存每次运行的模型和结果的目录
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
NGRAM_RANGE = (1, 3)

# 网格搜索参数
PARAM_TARGET_NUM_TOPICS = [12, 14, 16, 18, 20, 22, 24]
PARAM_UMAP_N_NEIGHBORS = [5, 10, 15]
PARAM_MIN_TOPIC_SIZE = [5, 10]

# 固定模型参数
UMAP_N_COMPONENTS = 5
UMAP_METRIC = 'cosine'
OUTLIER_REDUCTION_STRATEGY = "c-tf-idf"

# 用于跳过产生过多离群点参数组合的阈值
OUTLIER_SKIP_THRESHOLD = 0.55

# ----------------------------
# NLTK 初始化
# ----------------------------
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

# ----------------------------
# 停用词与预处理函数
# ----------------------------
CUSTOM_STOPWORDS_FILE = 'custom_stopwords.txt'
CUSTOM_PHRASES_FILE = 'custom_phrases.txt'

CUSTOM_PHRASES = []
if os.path.exists(CUSTOM_PHRASES_FILE):
    with open(CUSTOM_PHRASES_FILE, 'r', encoding='utf-8') as f:
        CUSTOM_PHRASES = sorted([line.strip().lower() for line in f if line.strip()], key=len, reverse=True)

custom_list = set()
if os.path.exists(CUSTOM_STOPWORDS_FILE):
    with open(CUSTOM_STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        custom_list = set([line.strip().lower() for line in f if line.strip()])

ENGLISH_STOPWORDS = set(stopwords.words('english'))
ENGLISH_STOPWORDS.update(custom_list)

LEMMA = WordNetLemmatizer()
ALLOWED_POS_TAGS = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def get_wordnet_pos(tag):
    """将POS词性标签映射到lemmatize()函数可接受的格式"""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    """对文本进行预处理，包括分词、词性标注、词形还原和去除停用词"""
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    # 将多词短语替换为单个token，以便将其视为一个实体
    for phrase in CUSTOM_PHRASES:
        text_lower = text_lower.replace(phrase, phrase.replace(' ', '_'))
    
    tokens = word_tokenize(text_lower)
    tagged = nltk.pos_tag(tokens)
    
    filtered_tokens = []
    for word, tag in tagged:
        if not word.isalnum() or tag not in ALLOWED_POS_TAGS:
            continue
        lemma_pos = get_wordnet_pos(tag)
        lemmatized_word = LEMMA.lemmatize(word, pos=lemma_pos)
        if lemmatized_word and lemmatized_word not in ENGLISH_STOPWORDS:
            filtered_tokens.append(lemmatized_word)
            
    return " ".join(filtered_tokens)

# ----------------------------
# Coherence 计算
# ----------------------------
def calculate_coherence_optimized(topic_model, processed_documents):
    """为训练好的 BERTopic 模型计算 C_V 一致性得分"""
    try:
        # 从模型中提取主题词
        topic_words = [
            [word[0] for word in topic_model.get_topic(topic_id)[:10]]
            for topic_id in topic_model.get_topics() if topic_id != -1
        ]
    except Exception:
        return 0.0

    if not topic_words:
        return 0.0

    texts = [doc.split() for doc in processed_documents if isinstance(doc, str) and doc.strip()]
    if not texts:
        return 0.0

    dictionary = Dictionary(texts)
    try:
        cm = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
        score = cm.get_coherence()
        return float(score) if score is not None else 0.0
    except Exception:
        return 0.0

# ----------------------------
# 单次实验运行
# ----------------------------
def run_bertopic_experiment(target_topics, n_neighbors, min_size, df_original, processed_documents, embeddings, embedding_model):
    """使用给定的参数集运行一次BERTopic实验"""
    start_time = time.time()
    param_str = f"T={target_topics}, N={n_neighbors}, S={min_size}"
    print(f"--- 运行实验: {param_str} ---")

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=UMAP_N_COMPONENTS,
        min_dist=0.0,
        metric=UMAP_METRIC,
        random_state=42
    )

    topic_model = BERTopic(
        language="english",
        min_topic_size=min_size,
        n_gram_range=NGRAM_RANGE,
        embedding_model=embedding_model,
        umap_model=umap_model,
        vectorizer_model=None,
        verbose=False
    )

    try:
        topics, probabilities = topic_model.fit_transform(processed_documents, embeddings=embeddings)
    except Exception as e:
        print(f"错误: fit_transform 失败: {e}")
        return {'TARGET_NUM_TOPICS': target_topics, 'UMAP_N_NEIGHBORS': n_neighbors, 'MIN_TOPIC_SIZE': min_size, 'Coherence_CV': 0.0, 'Error': f"fit_transform error: {e}"}

    num_topics_initial = len(topic_model.get_topic_info()) - 1
    outlier_count_initial = Counter(topics).get(-1, 0)
    outlier_ratio = outlier_count_initial / len(topics) if topics else 0

    if outlier_ratio > OUTLIER_SKIP_THRESHOLD:
        print(f"跳过: 离群点比例 {outlier_ratio:.2%} 超过阈值 {OUTLIER_SKIP_THRESHOLD:.2%}")
        return {'TARGET_NUM_TOPICS': target_topics, 'UMAP_N_NEIGHBORS': n_neighbors, 'MIN_TOPIC_SIZE': min_size, 'Coherence_CV': 0.0, 'Outlier_Ratio': outlier_ratio}

    # 阶段 1: 将话题减少到目标数量
    if num_topics_initial > target_topics:
        try:
            topic_model.reduce_topics(docs=processed_documents, nr_topics=target_topics)
            topics = topic_model.topics_
            probabilities = topic_model.probabilities_
        except Exception as e:
            print(f"警告: reduce_topics 失败，跳过话题合并。原因: {e}")

    # 阶段 2: 减少离群点
    if Counter(topics).get(-1, 0) > 0:
        topics = topic_model.reduce_outliers(processed_documents, topics, strategy=OUTLIER_REDUCTION_STRATEGY)

    # 最终同步话题状态
    try:
        topic_model.update_topics(docs=processed_documents, topics=topics)
    except Exception as e:
        print(f"警告: update_topics 失败: {e}")

    # 计算最终指标
    coherence_score = calculate_coherence_optimized(topic_model, processed_documents)
    num_topics_final = len(topic_model.get_topic_info()) - 1

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename_prefix = f"t{target_topics:02d}_n{n_neighbors:02d}_s{min_size:02d}"
    
    # 保存模型和可视化文件
    model_output_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_model")
    viz_output_file = os.path.join(OUTPUT_DIR, f"{filename_prefix}_visualization.html")
    try:
        topic_model.save(model_output_path, serialization="safetensors")
        fig = topic_model.visualize_topics()
        fig.write_html(viz_output_file)
    except Exception as e:
        print(f"警告: 保存模型或可视化文件失败: {e}")

    # 保存详细和汇总的 Excel 文件
    try:
        topic_info = topic_model.get_topic_info().rename(columns={'Topic': 'Topic_ID', 'Count': 'Topic_Count', 'Name': 'Topic_Keywords'})
        df_results = df_original.copy()
        df_results['Topic_ID'] = topics
        df_results = pd.merge(df_results, topic_info[['Topic_ID', 'Topic_Keywords']], on='Topic_ID', how='left')
        
        output_detail_file = os.path.join(OUTPUT_DIR, f"{filename_prefix}_detail.xlsx")
        output_summary_file = os.path.join(OUTPUT_DIR, f"{filename_prefix}_summary.xlsx")
        df_results.to_excel(output_detail_file, index=False)
        topic_info.to_excel(output_summary_file, index=False)
    except Exception as e:
        print(f"警告: 保存 Excel 结果失败: {e}")

    end_time = time.time()
    print(f"完成. 一致性 (C_V): {coherence_score:.4f} | 耗时: {end_time - start_time:.2f}s")

    return {
        'TARGET_NUM_TOPICS': target_topics,
        'UMAP_N_NEIGHBORS': n_neighbors,
        'MIN_TOPIC_SIZE': min_size,
        'Coherence_CV': coherence_score,
        'Outlier_Ratio': outlier_ratio,
        'Num_Topics_Initial': num_topics_initial,
        'Num_Topics_Final': num_topics_final
    }

# ----------------------------
# 主执行流程
# ----------------------------
def main():
    print("--- BERTopic 网格搜索初始化 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    try:
        df_original = pd.read_excel(FILE_PATH, usecols=['Article Title', 'Abstract'])
    except FileNotFoundError:
        print(f"错误: 输入文件 '{FILE_PATH}' 未找到。")
        return
    except Exception as e:
        print(f"错误: 读取输入文件失败: {e}")
        return

    df_original['Combined_Text'] = df_original['Article Title'].fillna('') + ". " + df_original['Abstract'].fillna('')
    documents = df_original['Combined_Text'].tolist()

    # 2. 文本预处理
    print("正在预处理文档...")
    processed_documents = [preprocess_text(doc) for doc in documents]
    empty_count = sum(1 for d in processed_documents if not d.strip())
    print(f"预处理完成。共发现 {empty_count} 个空文档 (总计 {len(documents)} 个)。")
    if empty_count / len(documents) > 0.4:
        print("警告: 超过 40% 的文档在预处理后为空。请检查停用词和词性过滤设置。")

    # 3. 加载嵌入模型并计算嵌入向量
    print(f"正在加载嵌入模型: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"错误: 无法加载嵌入模型 '{EMBEDDING_MODEL_NAME}': {e}")
        return

    print("正在计算文档嵌入向量 (此过程可能需要一些时间)...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    print("嵌入向量计算完成。")

    # 4. 网格搜索
    all_results = []
    param_combinations = [(t, n, s) for t in PARAM_TARGET_NUM_TOPICS for n in PARAM_UMAP_N_NEIGHBORS for s in PARAM_MIN_TOPIC_SIZE]
    total_runs = len(param_combinations)
    print(f"\n--- 开始网格搜索 (共 {total_runs} 种组合) ---")
    
    for i, (t, n, s) in enumerate(param_combinations):
        print(f"\n--- 运行第 {i+1}/{total_runs} 种组合 ---")
        result = run_bertopic_experiment(
            target_topics=t,
            n_neighbors=n,
            min_size=s,
            df_original=df_original,
            processed_documents=processed_documents,
            embeddings=embeddings,
            embedding_model=embedding_model
        )
        if result:
            all_results.append(result)

    # 5. 汇总并保存结果
    if not all_results:
        print("网格搜索完成，但没有生成任何结果。")
        return
        
    df_summary = pd.DataFrame(all_results)
    df_summary.sort_values(by='Coherence_CV', ascending=False, inplace=True)
    
    summary_output_file = os.path.join(OUTPUT_DIR, 'grid_search_coherence_summary.xlsx')
    df_summary.to_excel(summary_output_file, index=False)
    
    print(f"\n--- 网格搜索完成 ---")
    print(f"所有运行的汇总结果已保存至: {summary_output_file}")
    
    best_run = df_summary.iloc[0]
    print("\n--- 最佳运行结果摘要 ---")
    print(f"一致性 (C_V): {best_run['Coherence_CV']:.4f}")
    print(f"参数: T={best_run['TARGET_NUM_TOPICS']}, N={best_run['UMAP_N_NEIGHBORS']}, S={best_run['MIN_TOPIC_SIZE']}")
    print(f"最终话题数: {best_run['Num_Topics_Final']}")
    print(f"此运行的详细结果保存在 '{OUTPUT_DIR}' 目录中。")


if __name__ == '__main__':
    main()
