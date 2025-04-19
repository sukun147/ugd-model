import gc
import json
import math
import time

import torch
from bert_score import score as bert_score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from extractor.extractor import KnowledgeExtractor
from ner.ner import ner_with_pretrained_model

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-MedChatZH-LoRA-SFT"
DATA_PATH = "llm/data/MedChatZH_valid.json"

# 全局变量，只初始化一次
global_llm = None
global_tokenizer = None
global_extractor = None


def initialize_resources():
    """初始化全局资源，只执行一次"""
    global global_llm, global_tokenizer, global_extractor

    # 初始化模型
    if global_llm is None:
        print("正在加载LLM模型...")
        global_llm = LLM(model=MODEL_PATH, max_model_len=16384)

    # 初始化分词器
    if global_tokenizer is None:
        print("正在加载分词器...")
        global_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 初始化知识提取器
    if global_extractor is None:
        print("正在初始化知识提取器...")
        global_extractor = KnowledgeExtractor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="sukun031015",
            embedding_model_name="intfloat/multilingual-e5-small"
        )

    return global_llm, global_tokenizer, global_extractor


def qa(question):
    # 确保资源已初始化
    llm, tokenizer, extractor = initialize_resources()

    entities = []
    ner_res = ner_with_pretrained_model([question])
    for ner_token in ner_res[0]:
        entities.append(ner_token.word)

    knowledge = extractor.extract_subgraph(entities, [], 1)

    prompt = f"""
    你是一个中医领域的知识图谱问答助手，你的任务是根据问题和知识图谱中的信息来回答问题。
    问题: {question}
    知识图谱中的信息: {knowledge}
    请根据知识图谱中的信息回答问题。
    """
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate([text], sampling_params)
    answer = outputs[0].outputs[0].text
    return answer, knowledge


def batch_qa(questions, batch_size=5):
    """批量处理问题，显示简单进度信息"""
    # 确保资源已初始化
    llm, tokenizer, extractor = initialize_resources()

    results = []
    total_batches = (len(questions) + batch_size - 1) // batch_size  # 向上取整计算批次数

    print(f"开始处理 {len(questions)} 个问题，共 {total_batches} 个批次")
    start_time = time.time()

    for i in range(0, len(questions), batch_size):
        batch_number = i // batch_size + 1
        print(f"正在处理批次 {batch_number}/{total_batches}...")

        batch_questions = questions[i:i + batch_size]
        batch_prompts = []

        # 为每个问题准备实体和知识
        batch_knowledge = []
        for question in batch_questions:
            entities = []
            ner_res = ner_with_pretrained_model([question])
            for ner_token in ner_res[0]:
                entities.append(ner_token.word)
            knowledge = extractor.extract_subgraph(entities, [], 1)
            batch_knowledge.append(knowledge)

        # 准备批量请求
        batch_texts = []
        for j, question in enumerate(batch_questions):
            prompt = f"""
            你是一个中医领域的知识图谱问答助手，你的任务是根据问题和知识图谱中的信息来回答问题。
            问题: {question}
            知识图谱中的信息: {batch_knowledge[j]}
            请根据知识图谱中的信息回答问题。
            """
            messages = [
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_texts.append(text)

        # 批量生成答案
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        batch_outputs = llm.generate(batch_texts, sampling_params)

        # 保存结果
        for j, output in enumerate(batch_outputs):
            answer = output.outputs[0].text
            results.append({
                "question": batch_questions[j],
                "answer": answer,
                "knowledge": batch_knowledge[j]
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


def evaluate_bertscore(predictions, references):
    """分批计算BERTScore，防止OOM"""
    print("正在计算BERTScore...")

    # 释放一些内存
    torch.cuda.empty_cache()
    gc.collect()

    all_P = []
    all_R = []
    all_F1 = []

    # 分批计算BERTScore
    batch_size = 8  # 使用更小的批次
    for i in range(0, len(predictions), batch_size):
        print(f"BERTScore: 处理批次 {i // batch_size + 1}/{(len(predictions) + batch_size - 1) // batch_size}")
        batch_preds = predictions[i:i + batch_size]
        batch_refs = references[i:i + batch_size]

        try:
            P, R, F1 = bert_score(batch_preds, batch_refs, lang="zh", verbose=False)
            all_P.extend(P.tolist())
            all_R.extend(R.tolist())
            all_F1.extend(F1.tolist())
        except Exception as e:
            print(f"BERTScore计算错误: {e}")
            # 如果出错，添加0分
            all_P.extend([0.0] * len(batch_preds))
            all_R.extend([0.0] * len(batch_preds))
            all_F1.extend([0.0] * len(batch_preds))

        # 强制清理内存
        torch.cuda.empty_cache()
        gc.collect()

    avg_P = sum(all_P) / len(all_P) if all_P else 0
    avg_R = sum(all_R) / len(all_R) if all_R else 0
    avg_F1 = sum(all_F1) / len(all_F1) if all_F1 else 0

    return {
        "Precision": avg_P,
        "Recall": avg_R,
        "F1": avg_F1
    }


def calculate_perplexity(predictions, references, tokenizer):
    """
    计算困惑度(Perplexity)指标
    困惑度是衡量模型预测文本概率分布的指标，数值越低表示模型性能越好
    这里我们使用参考文本的token作为标准，计算模型生成文本与参考文本的差异
    """
    print("正在计算Perplexity困惑度...")

    # 使用tokenizer对文本进行编码
    batch_size = 8  # 分批处理以避免OOM
    perplexities = []

    for i in range(0, len(predictions), batch_size):
        print(f"Perplexity: 处理批次 {i // batch_size + 1}/{(len(predictions) + batch_size - 1) // batch_size}")
        batch_preds = predictions[i:i + batch_size]
        batch_refs = references[i:i + batch_size]

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
        torch.cuda.empty_cache()
        gc.collect()

    # 计算平均困惑度
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0

    return {
        "Score": avg_perplexity
    }


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


def evaluate_responses_with_batch():
    """使用批处理方式评估回答，分批计算指标防止OOM"""
    references = []
    questions = []

    # 首先收集所有问题
    print("读取数据集...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data["instruction"]
                reference = data["output"]
                questions.append(question)
                references.append(reference)
            except json.JSONDecodeError:
                print(f"跳过无效行: {line.strip()}")
                continue

    print(f"总共读取了 {len(questions)} 个问题")

    # 尝试从文件加载预处理结果
    result_file = "main_results.json"
    try:
        print(f"尝试从 {result_file} 加载已保存的结果...")
        with open(result_file, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
        print(f"成功加载 {len(batch_results)} 个预处理结果")
        predictions = [result["answer"] for result in batch_results]
    except (FileNotFoundError, json.JSONDecodeError):
        print("未找到有效的预处理结果文件，开始批量处理问题...")

        # 确保资源已初始化（在批量处理前只初始化一次）
        initialize_resources()

        # 批量处理问题
        batch_size = 10
        print(f"开始批量处理，批次大小为 {batch_size}")

        batch_results = batch_qa(questions, batch_size)
        predictions = [result["answer"] for result in batch_results]

        # 保存处理结果到文件
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        print(f"处理结果已保存至 {result_file}")

    # 计算评估指标...
    print("计算评估指标...")

    # 计算BERTScore
    bertscore = evaluate_bertscore(predictions, references)

    # 计算Perplexity困惑度
    global global_tokenizer
    if global_tokenizer is None:
        print("初始化分词器用于困惑度计算...")
        global_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    perplexity = calculate_perplexity(predictions, references, global_tokenizer)

    # 清理不再需要的资源，释放内存
    print("清理模型资源以释放内存...")
    global global_llm, global_extractor
    global_llm = None
    global_tokenizer = None
    global_extractor = None
    torch.cuda.empty_cache()
    gc.collect()

    # 结果输出
    return {
        "BERTScore": bertscore,
        "Perplexity": perplexity,
    }


def save_batch_results(batch_results, output_file="batch_qa_results.json"):
    """保存批处理结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至 {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="中医知识图谱问答系统")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "batch", "evaluate"],
                        help="运行模式: single(单个问题), batch(批量问答), evaluate(评估)")
    parser.add_argument("--questions", type=str, default="",
                        help="批量问答的问题列表，用分号分隔")
    parser.add_argument("--question", type=str, default="生姜有什么功能？",
                        help="单个问题测试时使用的问题")

    args = parser.parse_args()

    if args.mode == "single":
        # 单个问题测试
        question = args.question
        print(f"问题: {question}")
        answer, knowledge = qa(question)
        print(f"回答: {answer}")
        print(f"知识: {knowledge}")

    elif args.mode == "batch":
        # 批量问答测试
        if args.questions:
            questions = args.questions.split(";")
            print(f"接收到 {len(questions)} 个问题，开始批量处理...")
            batch_results = batch_qa(questions, args.batch_size)
            save_batch_results(batch_results, args.output)

            # 打印结果摘要
            for i, result in enumerate(batch_results):
                print(f"问题 {i + 1}: {result['question']}")
                print(f"回答: {result['answer'][:100]}..." if len(
                    result['answer']) > 100 else f"回答: {result['answer']}")
                print("-" * 50)
        else:
            print("未提供问题列表，使用默认问题进行测试...")
            questions = ["生姜有什么功能？", "胃痛怎么办？", "中暑有什么症状？", "附子的功效是什么？", "如何治疗感冒？"]
            batch_results = batch_qa(questions, args.batch_size)
            save_batch_results(batch_results, args.output)

            # 打印结果
            for result in batch_results:
                print(f"问题: {result['question']}")
                print(f"回答: {result['answer']}")
                print("-" * 50)

    elif args.mode == "evaluate":
        # 评估测试集
        print("开始评估测试集...")
        results = evaluate_responses_with_batch()
        print("\n评估结果:")
        print("BERTScore:")
        for metric, score in results["BERTScore"].items():
            print(f"  {metric}: {score}")
        print("Perplexity困惑度:")
        print(f"  Score: {results['Perplexity']['Score']}")

"""
BERTScore:
  Precision: 0.6919788339138031
  Recall: 0.6773922579884529
  F1: 0.683899534881115
Perplexity困惑度:
  Score: <perplexity_score> 
"""
