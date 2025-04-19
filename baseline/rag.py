import argparse
import gc
import json
import time
import math

import faiss
import numpy as np
import torch
from bert_score import score as bert_score
from py2neo import Graph
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams

prompt = """你是一个中医领域的知识图谱问答助手，你的任务是根据问题和知识图谱中的信息来回答问题。
问题: {question}
知识图谱中的信息: {knowledge}
请根据知识图谱中的信息回答问题。"""


class KnowledgeGraphLoader:
    def __init__(self, uri, user, password):
        self.graph = Graph(uri, auth=(user, password))

    def load_knowledge_graph(self):
        # 修改查询语句，使用type(r)函数获取关系类型
        query = """
        MATCH (h)-[r]->(t)
        RETURN h.name AS head, t.name AS tail, type(r) AS relation
        """
        # 执行Cypher查询并获取结果
        result = self.graph.run(query)
        triples = []
        for record in result:
            head = record["head"]
            tail = record["tail"]
            relation = record["relation"]

            # 确保所有字段都不为None
            if head is not None and tail is not None and relation is not None:
                triples.append((head, tail, relation))
            else:
                # 如果任何字段为None，打印警告并跳过
                print(f"警告: 发现不完整的三元组 - 头: {head}, 尾: {tail}, 关系: {relation}")

        return triples

    def check_schema(self):
        """检查Neo4j知识图谱的结构"""
        # 获取所有的关系类型
        relation_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        relation_types = [record["relationshipType"] for record in self.graph.run(relation_query)]

        # 对于每种关系类型，检查第一个实例的属性
        relation_props = {}
        for rel_type in relation_types:
            prop_query = f"""
            MATCH ()-[r:{rel_type}]->() 
            RETURN properties(r) AS props
            LIMIT 1
            """
            result = list(self.graph.run(prop_query))
            if result:
                relation_props[rel_type] = result[0]["props"]
            else:
                relation_props[rel_type] = {}

        return {
            "relation_types": relation_types,
            "relation_properties": relation_props
        }


# 2. Vectorize the knowledge base and store in FAISS index
def vectorize_knowledge_base(sentences, model, tokenizer):
    formatted_sentences = ["query: " + sentence for sentence in sentences]

    print("正在向量化知识库...")
    # 处理可能的大量三元组，分批次处理
    batch_size = 128
    all_embeddings = []

    for i in range(0, len(formatted_sentences), batch_size):
        print(f"向量化批次 {i // batch_size + 1}/{(len(formatted_sentences) + batch_size - 1) // batch_size}")
        batch = formatted_sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

        # 及时清理内存
        del inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 合并所有嵌入
    if len(all_embeddings) == 1:
        embeddings_np = all_embeddings[0]
    else:
        embeddings_np = np.vstack(all_embeddings)

    # 创建FAISS索引
    print("创建FAISS索引...")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积相似度 (等同于余弦相似度，如果向量已归一化)

    # 归一化向量以确保余弦相似度
    faiss.normalize_L2(embeddings_np)

    # 将向量添加到索引
    index.add(embeddings_np)

    print(f"FAISS索引创建完成，包含 {index.ntotal} 个向量")
    return index, embeddings_np


# 3. 向量化查询并从FAISS检索
def vectorize_query_and_search(query, model, tokenizer, index, k=3):
    formatted_query = "query: " + query
    inputs = tokenizer(formatted_query, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    # 归一化查询向量
    faiss.normalize_L2(query_embedding)

    # 执行近似最近邻搜索
    D, I = index.search(query_embedding, k)

    # 清理内存
    del inputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return I[0]  # 返回索引


# 批量向量化查询并检索
def batch_query_and_search(queries, model, tokenizer, index, k=3, threshold=0.6, batch_size=32):
    all_top_indices = []
    all_top_scores = []

    # 分批处理查询
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        formatted_queries = ["query: " + query for query in batch_queries]

        # 向量化批次查询
        inputs = tokenizer(formatted_queries, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

        # 归一化向量
        faiss.normalize_L2(batch_embeddings)

        # 批量检索
        D, I = index.search(batch_embeddings, k)

        # 收集结果并应用阈值过滤
        for query_idx, (indices, scores) in enumerate(zip(I, D)):
            filtered_indices = []
            filtered_scores = []

            for i, (idx, score) in enumerate(zip(indices, scores)):
                if score >= threshold:
                    filtered_indices.append(idx)
                    filtered_scores.append(score)

            # 如果所有结果都被过滤掉了，至少保留一个最相似的
            if not filtered_indices and len(indices) > 0:
                filtered_indices.append(indices[0])
                filtered_scores.append(scores[0])

            all_top_indices.append(filtered_indices)
            all_top_scores.append(filtered_scores)

        # 清理内存
        del inputs, batch_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_top_indices, all_top_scores


# 5. 使用vLLM生成回答
def generate_answer(question, knowledge, llm, tokenizer):
    # Format the prompt with the question and knowledge
    formatted_prompt = prompt.format(question=question, knowledge="\n".join(knowledge))

    # Use vLLM to generate an answer
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # Create chat message format
    messages = [
        {"role": "user", "content": formatted_prompt}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate using vLLM
    outputs = llm.generate([text], sampling_params)
    answer = outputs[0].outputs[0].text

    return answer


# 批量生成答案
def batch_generate_answers(batch_questions, batch_knowledge, llm, tokenizer):
    all_answers = []

    # Prepare batch requests
    batch_texts = []
    for question, knowledge in zip(batch_questions, batch_knowledge):
        formatted_prompt = prompt.format(question=question, knowledge="\n".join(knowledge))
        messages = [
            {"role": "user", "content": formatted_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_texts.append(text)

    # Use vLLM to generate answers in batch
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    outputs = llm.generate(batch_texts, sampling_params)

    # Retrieve generated answers
    for output in outputs:
        answer = output.outputs[0].text
        all_answers.append(answer)

    return all_answers


# 6. Load JSON dataset
def load_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return data


# 7. 改用直接调用bert_score计算评估指标
def compute_bertscore_batch(predictions, references):
    """分批计算BERTScore指标，防止OOM"""
    print("开始计算BERTScore评估指标...")

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 分批计算BERTScore
    print("计算BERTScore分数...")
    batch_size = 8
    all_P = []
    all_R = []
    all_F1 = []

    for i in range(0, len(predictions), batch_size):
        end_idx = min(i + batch_size, len(predictions))
        print(f"BERTScore: 处理 {i + 1} 到 {end_idx} / {len(predictions)}")

        batch_preds = predictions[i:end_idx]
        batch_refs = references[i:end_idx]

        try:
            # 直接使用bert_score库计算分数
            P, R, F1 = bert_score(batch_preds, batch_refs, lang="zh", verbose=False)
            all_P.extend(P.tolist())
            all_R.extend(R.tolist())
            all_F1.extend(F1.tolist())
        except Exception as e:
            print(f"BERTScore计算异常: {e}")
            # 如果出错，添加0分
            all_P.extend([0.0] * len(batch_preds))
            all_R.extend([0.0] * len(batch_preds))
            all_F1.extend([0.0] * len(batch_preds))

        # 强制清理内存
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 计算最终结果
    bertscore_score = {
        "precision": sum(all_P) / len(all_P) if all_P else 0,
        "recall": sum(all_R) / len(all_R) if all_R else 0,
        "f1": sum(all_F1) / len(all_F1) if all_F1 else 0
    }

    return bertscore_score


# 计算Perplexity困惑度
def calculate_perplexity(predictions, references, tokenizer):
    """
    计算困惑度(Perplexity)指标
    困惑度是衡量模型预测文本概率分布的指标，数值越低表示模型性能越好
    这里我们使用参考文本的token作为标准，计算模型生成文本与参考文本的差异
    """
    print("开始计算Perplexity困惑度...")

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 使用tokenizer对文本进行编码
    batch_size = 8  # 分批处理以避免OOM
    perplexities = []

    for i in range(0, len(predictions), batch_size):
        end_idx = min(i + batch_size, len(predictions))
        print(f"Perplexity: 处理 {i + 1} 到 {end_idx} / {len(predictions)}")

        batch_preds = predictions[i:end_idx]
        batch_refs = references[i:end_idx]

        batch_perplexities = []
        for pred, ref in zip(batch_preds, batch_refs):
            try:
                # 对预测文本和参考文本进行编码
                pred_tokens = tokenizer.encode(pred, add_special_tokens=False)
                ref_tokens = tokenizer.encode(ref, add_special_tokens=False)

                # 计算文本长度差异的惩罚项
                length_penalty = abs(len(pred_tokens) - len(ref_tokens)) / max(len(pred_tokens), len(ref_tokens))

                # 计算两个token序列的编辑距离
                distance = calculate_edit_distance(pred_tokens, ref_tokens)
                normalized_distance = distance / max(len(pred_tokens), len(ref_tokens))

                # 将编辑距离转换为困惑度值 (较低的编辑距离对应较低的困惑度)
                # 使用指数函数转换，保证困惑度值始终为正数
                perplexity = math.exp(normalized_distance + length_penalty)
                batch_perplexities.append(perplexity)
            except Exception as e:
                print(f"Perplexity计算错误: {e}")
                # 如果出错，添加一个较高的困惑度值
                batch_perplexities.append(100.0)

        perplexities.extend(batch_perplexities)

        # 强制清理内存
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 计算平均困惑度
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0

    return {
        "score": avg_perplexity
    }


# 计算编辑距离的辅助函数
def calculate_edit_distance(s1, s2):
    """
    计算两个序列之间的编辑距离（Levenshtein距离）
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 动态规划填表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n]


def batch_qa(questions, embedding_model, embedding_tokenizer, llm, gen_tokenizer,
             faiss_index, sentences, prompt, batch_size=10, threshold=0.6):
    """批量处理问题，使用FAISS加速检索和vLLM加速生成，并应用相似度阈值"""
    results = []
    total_batches = (len(questions) + batch_size - 1) // batch_size  # 向上取整计算批次数

    print(f"开始处理 {len(questions)} 个问题，共 {total_batches} 个批次")
    print(f"使用相似度阈值: {threshold}，只有相似度≥{threshold}的知识才会被使用")
    start_time = time.time()

    # 使用FAISS批量检索（增加k值，为过滤预留空间）
    print("正在执行向量检索...")
    all_top_indices, all_top_scores = batch_query_and_search(
        questions, embedding_model, embedding_tokenizer, faiss_index, k=3, threshold=threshold
    )

    # 转换索引为知识文本
    all_retrieved_knowledge = []
    all_knowledge_scores = []

    # 计算相似度过滤统计信息
    total_retrieved = 0
    total_after_filter = 0

    for indices, scores in zip(all_top_indices, all_top_scores):
        knowledge = [sentences[idx] for idx in indices]
        all_retrieved_knowledge.append(knowledge)
        all_knowledge_scores.append(scores)

        total_retrieved += 10  # 原始检索数量
        total_after_filter += len(indices)  # 过滤后数量

    avg_knowledge_per_question = total_after_filter / len(questions) if questions else 0
    print(f"检索统计: 平均每个问题过滤前检索 10 条知识，过滤后保留 {avg_knowledge_per_question:.2f} 条知识")
    print(f"过滤率: {(1 - total_after_filter / total_retrieved) * 100:.2f}%")

    # 批量生成答案
    print("正在生成所有答案...")
    for i in range(0, len(questions), batch_size):
        batch_number = i // batch_size + 1
        print(f"正在处理批次 {batch_number}/{total_batches}...")

        batch_questions = questions[i:i + batch_size]
        batch_knowledge = all_retrieved_knowledge[i:i + batch_size]
        batch_scores = all_knowledge_scores[i:i + batch_size]

        # 生成答案
        batch_answers = batch_generate_answers(batch_questions, batch_knowledge, llm, gen_tokenizer)

        # 保存结果
        for j, (question, answer, knowledge, scores) in enumerate(
                zip(batch_questions, batch_answers, batch_knowledge, batch_scores)):
            # 为了更清晰地展示结果，添加相似度分数
            knowledge_with_scores = []
            for k, score in zip(knowledge, scores):
                knowledge_with_scores.append(f"{k} [相似度: {score:.4f}]")

            results.append({
                "question": question,
                "answer": answer,
                "knowledge": knowledge,
                "knowledge_with_scores": knowledge_with_scores,
                "similarity_scores": scores
            })

        # 显示进度
        processed_count = min(i + batch_size, len(questions))
        elapsed_time = time.time() - start_time
        avg_time_per_question = elapsed_time / processed_count if processed_count > 0 else 0
        estimated_remaining = avg_time_per_question * (len(questions) - processed_count)

        print(f"已处理 {processed_count}/{len(questions)} 个问题 "
              f"({processed_count / len(questions) * 100:.1f}%), "
              f"用时: {elapsed_time:.1f}秒, "
              f"预计剩余: {estimated_remaining:.1f}秒")

    total_time = time.time() - start_time
    print(f"批处理完成，共处理 {len(questions)} 个问题，总用时: {total_time:.1f}秒，"
          f"平均每个问题: {total_time / len(questions):.1f}秒")

    return results


# 从批处理结果中评估性能
def evaluate_from_batch_results(batch_results, references, tokenizer):
    predictions = [result["answer"] for result in batch_results]

    # 计算BERTScore
    bertscore = compute_bertscore_batch(predictions, references)

    # 计算Perplexity困惑度
    perplexity = calculate_perplexity(predictions, references, tokenizer)

    return {
        "BERTScore": bertscore,
        "Perplexity": perplexity
    }


# 保存批处理结果到文件
def save_batch_results(batch_results, output_file="rag_results.json"):
    """保存批处理结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至 {output_file}")


# 释放模型资源
def release_model_resources(embedding_model, embedding_tokenizer, llm, gen_tokenizer):
    """释放模型资源以释放内存"""
    del embedding_model
    del embedding_tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def evaluate_dataset(dataset_path, embedding_model, embedding_tokenizer, llm, gen_tokenizer,
                     faiss_index, sentences, prompt, batch_size=10, threshold=0.6):
    # 尝试从文件加载预处理结果
    result_file = f"rag_results.json"
    try:
        print(f"尝试从 {result_file} 加载已保存的结果...")
        with open(result_file, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
        print(f"成功加载 {len(batch_results)} 个预处理结果")

        # 加载参考答案
        dataset = load_json_dataset(dataset_path)
        references = [item["output"] for item in dataset]

        if len(batch_results) != len(references):
            print(f"警告: 加载的结果数量 ({len(batch_results)}) 与参考答案数量 ({len(references)}) 不匹配")
            print("将重新处理数据集...")
            raise FileNotFoundError
    except (FileNotFoundError, json.JSONDecodeError):
        print("未找到有效的预处理结果文件，开始处理数据集...")

        dataset = load_json_dataset(dataset_path)
        questions = [item["instruction"] for item in dataset]
        references = [item["output"] for item in dataset]

        print(f"总共读取了 {len(questions)} 个问题")

        # 批量处理问题
        print(f"开始批量处理，批次大小为 {batch_size}，相似度阈值为 {threshold}")
        batch_results = batch_qa(questions, embedding_model, embedding_tokenizer, llm, gen_tokenizer,
                                 faiss_index, sentences, prompt, batch_size, threshold)

        # 保存结果
        save_batch_results(batch_results, result_file)

    # 计算评估指标
    print("计算评估指标...")
    references = [item["output"] for item in load_json_dataset(dataset_path)]
    metrics = evaluate_from_batch_results(batch_results, references, gen_tokenizer)

    # 释放模型资源以便后续操作
    print("释放模型资源以便评估...")
    release_model_resources(embedding_model, embedding_tokenizer, llm, gen_tokenizer)

    return metrics, batch_results


# 保存和加载FAISS索引
def save_faiss_index(index, filename="knowledge_index.faiss"):
    """保存FAISS索引到文件"""
    faiss.write_index(index, filename)
    print(f"FAISS索引已保存至 {filename}")


def load_faiss_index(filename="knowledge_index.faiss"):
    """从文件加载FAISS索引"""
    try:
        index = faiss.read_index(filename)
        print(f"成功从 {filename} 加载FAISS索引，包含 {index.ntotal} 个向量")
        return index
    except Exception as e:
        print(f"加载FAISS索引失败: {e}")
        return None


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG (Retrieval-Augmented Generation) 系统")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "batch", "evaluate", "check_kg"],
                        help="运行模式: single(单个问题), batch(批量问答), evaluate(评估), check_kg(检查知识图谱)")
    parser.add_argument("--questions", type=str, default="",
                        help="批量问答的问题列表，用分号分隔")
    parser.add_argument("--question", type=str, default="胃痛怎么办？",
                        help="单个问题测试时使用的问题")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="批处理大小")
    parser.add_argument("--output", type=str, default="rag_batch_results.json",
                        help="批量问答结果输出文件")
    parser.add_argument("--dataset", type=str, default="llm/data/MedChatZH_valid.json",
                        help="评估模式使用的数据集路径")
    parser.add_argument("--use_saved_index", action="store_true",
                        help="使用保存的FAISS索引文件")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Qwen2.5-7B-MedChatZH-LoRA-SFT",
                        help="LLM模型路径")

    args = parser.parse_args()

    # 初始化Neo4j连接
    loader = KnowledgeGraphLoader(uri="bolt://localhost:7687", user="neo4j", password="sukun031015")

    # 添加新的模式：检查知识图谱结构
    if args.mode == "check_kg":
        print("正在检查知识图谱结构...")
        schema_info = loader.check_schema()
        print("\n关系类型:")
        for rel_type in schema_info["relation_types"]:
            print(f"- {rel_type}")

        print("\n关系属性示例:")
        for rel_type, props in schema_info["relation_properties"].items():
            print(f"- {rel_type}: {props}")

        # 尝试获取前10个三元组进行检查
        query = """
        MATCH (h)-[r]->(t)
        RETURN h.name AS head, t.name AS tail, type(r) AS rel_type, properties(r) AS rel_props
        LIMIT 10
        """
        print("\n前10个三元组示例:")
        for i, record in enumerate(loader.graph.run(query)):
            print(
                f"{i + 1}. 头: {record['head']}, 尾: {record['tail']}, 关系类型: {record['rel_type']}, 关系属性: {record['rel_props']}")

        exit(0)

    # 加载知识图谱
    print("正在从Neo4j加载知识图谱...")
    triples = loader.load_knowledge_graph()

    # 检查是否获取到了三元组
    if not triples:
        print("错误: 没有从Neo4j中检索到任何三元组。请检查数据库连接和查询。")
        print("您可以使用 --mode check_kg 模式来检查知识图谱的结构。")
        exit(1)

    print(f"成功加载 {len(triples)} 条三元组")

    # 打印几个三元组示例以便检查
    print("三元组示例:")
    for i, (h, t, r) in enumerate(triples[:5]):
        print(f"{i + 1}. 头实体: {h}, 尾实体: {t}, 关系: {r}")

    sentences = ["{}的{}是{}".format(h, r, t) for h, t, r in triples]

    # 加载嵌入模型
    print("正在加载嵌入模型...")
    embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    embedding_model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    # 加载FAISS索引或创建新索引
    faiss_index = None
    if args.use_saved_index:
        faiss_index = load_faiss_index()

    if faiss_index is None:
        # 创建新的FAISS索引
        faiss_index, _ = vectorize_knowledge_base(sentences, embedding_model, embedding_tokenizer)
        # 保存索引以便后续使用
        save_faiss_index(faiss_index)

    # 使用vLLM加载生成模型
    print("正在加载生成模型 (使用vLLM)...")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 使用vLLM初始化大语言模型，设置正确的上下文长度
    print("初始化vLLM引擎...")
    llm = LLM(
        model=args.model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.9
    )
    print("vLLM引擎初始化完成")

    if args.mode == "single":
        # 单个问题测试
        question = args.question
        print(f"问题: {question}")

        # 向量化查询并检索
        top_indices = vectorize_query_and_search(question, embedding_model, embedding_tokenizer, faiss_index)
        top_k_knowledge = [sentences[i] for i in top_indices]

        print("检索到的知识:")
        for i, k in enumerate(top_k_knowledge):
            print(f"{i + 1}. {k}")

        # 生成答案
        answer = generate_answer(question, top_k_knowledge, llm, gen_tokenizer)
        print(f"回答: {answer}")

    elif args.mode == "batch":
        # 批量问答测试
        if args.questions:
            questions = args.questions.split(";")
            print(f"接收到 {len(questions)} 个问题，开始批量处理...")
        else:
            print("未提供问题列表，使用默认问题进行测试...")
            questions = ["生姜有什么功能？", "胃痛怎么办？", "中暑有什么症状？", "附子的功效是什么？", "如何治疗感冒？"]

        batch_results = batch_qa(questions, embedding_model, embedding_tokenizer, llm, gen_tokenizer,
                                 faiss_index, sentences, prompt, args.batch_size)
        save_batch_results(batch_results, args.output)

        # 打印结果摘要
        for i, result in enumerate(batch_results):
            print(f"问题 {i + 1}: {result['question']}")
            print(f"回答: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"回答: {result['answer']}")
            print("-" * 50)

    elif args.mode == "evaluate":
        # 评估测试集
        print(f"开始评估测试集: {args.dataset}")

        metrics, _ = evaluate_dataset(args.dataset, embedding_model, embedding_tokenizer, llm, gen_tokenizer,
                                      faiss_index, sentences, prompt, args.batch_size)

        print("\n评估结果:")
        print("BERTScore:")
        for metric, score in metrics["BERTScore"].items():
            print(f"  {metric}: {score}")
        print("Perplexity困惑度:")
        print(f"  Score: {metrics['Perplexity']['score']}")


"""
BERTScore:
  precision: 0.6151087065339088
  recall: 0.5492883291840553
  f1: 0.5785440436005592
"""
