import argparse
import gc
import json
import time

import torch
from bert_score import score as bert_score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# 使用vLLM生成回答
def generate_answer(prompt, llm, tokenizer):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # 创建聊天消息格式
    messages = [{"role": "user", "content": prompt}]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 使用vLLM生成回答
    outputs = llm.generate([text], sampling_params)
    answer = outputs[0].outputs[0].text

    return answer


# 批量问答生成答案
def batch_vllm_qa(questions, llm, tokenizer, batch_size=5):
    """
    批量处理问题，使用vLLM生成答案
    """
    results = []
    total_batches = (len(questions) + batch_size - 1) // batch_size  # 计算总批次数

    print(f"开始批量问答，共有 {len(questions)} 个问题，分为 {total_batches} 个批次")
    start_time = time.time()

    for i in range(0, len(questions), batch_size):
        batch_number = i // batch_size + 1
        print(f"正在处理第 {batch_number}/{total_batches} 批次...")

        # 当前批次的问题
        batch_questions = questions[i:i + batch_size]

        # 构建批次的prompt
        batch_prompts = []
        for question in batch_questions:
            prompt = f"你是一个中医领域的知识问答助手。以下是用户的问题：\n{question}\n请给出详细的回答。"
            batch_prompts.append(prompt)

        # 使用vLLM生成答案
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
        outputs = llm.generate(batch_prompts, sampling_params)

        # 收集答案
        for question, output in zip(batch_questions, outputs):
            answer = output.outputs[0].text
            results.append({
                "question": question,
                "answer": answer
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


# 保存批量问答结果
def save_batch_qa_results(results, output_file="vllm_batch_qa_results.json"):
    """
    保存批量问答的结果到文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至 {output_file}")


# 加载JSON数据集
def load_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return data


# 计算BERTScore评估指标
def compute_bertscore(predictions, references):
    print("开始计算BERTScore评估指标...")
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 分批计算BERTScore
    batch_size = 8  # 小批次以降低显存需求
    all_P, all_R, all_F1 = [], [], []

    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i + batch_size]
        batch_refs = references[i:i + batch_size]

        try:
            P, R, F1 = bert_score(batch_preds, batch_refs, lang="zh", verbose=False)
            all_P.extend(P.tolist())
            all_R.extend(R.tolist())
            all_F1.extend(F1.tolist())
        except Exception as e:
            print(f"BERTScore计算异常: {e}")
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


# 批量评估问答结果
def batch_evaluate(questions, references, llm, tokenizer, batch_size=10):
    """
    批量评估问答结果
    :param questions: 问题列表
    :param references: 参考答案列表
    :param llm: vLLM模型实例
    :param tokenizer: tokenizer实例
    :param batch_size: 批处理大小
    :return: 评估结果（如BERTScore）
    """
    # 生成答案
    print("开始批量生成答案...")
    predictions = []
    perplexities = []  # 存储每个回答的困惑度
    total_batches = (len(questions) + batch_size - 1) // batch_size  # 计算总批次数

    for i in range(0, len(questions), batch_size):
        batch_number = i // batch_size + 1
        print(f"正在生成第 {batch_number}/{total_batches} 批次...")

        batch_questions = questions[i:i + batch_size]

        # 构建批次的prompt
        batch_prompts = []
        for question in batch_questions:
            prompt = f"你是一个中医领域的知识问答助手。以下是用户的问题：\n{question}\n请给出详细的回答。"
            batch_prompts.append(prompt)

        # 使用vLLM生成答案(开启logprobs用于计算困惑度)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, logprobs=5)
        outputs = llm.generate(batch_prompts, sampling_params)

        # 收集答案和计算困惑度
        for output in outputs:
            predictions.append(output.outputs[0].text)
            # 计算困惑度
            logprobs = output.outputs[0].logprobs
            if logprobs:  # 确保有logprobs数据
                total_logprob = sum(logprob[0].logprob for logprob in logprobs if logprob)
                perplexity = torch.exp(torch.tensor(-total_logprob / len(logprobs))).item()
                perplexities.append(perplexity)

    # 计算BERTScore
    print("开始计算BERTScore...")
    bertscore_results = compute_bertscore(predictions, references)

    # 计算平均困惑度
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('inf')
    print(f"平均困惑度: {avg_perplexity:.2f}")

    return {
        "predictions": predictions,
        "bertscore": bertscore_results,
        "perplexity": avg_perplexity
    }


# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于vLLM的问答Baseline")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "evaluate"],
                        help="运行模式: single(单个问题), evaluate(评估)")
    parser.add_argument("--question", type=str, default="生姜的作用是？", help="单个问题测试时使用的问题")
    parser.add_argument("--dataset", type=str, default="llm/data/MedChatZH_valid.json", help="评估模式使用的数据集路径")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="results.json", help="输出结果文件路径")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")

    args = parser.parse_args()

    # 初始化生成模型
    print("加载vLLM模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.9
    )
    print("vLLM模型加载完成")

    if args.mode == "single":
        # 单个问题问答
        question = args.question
        print(f"问题: {question}")

        # 构建prompt
        prompt = f"你是一个中医问诊专家。以下是用户的问题：\n{question}\n请给出详细的回答。"

        # 生成答案
        answer = generate_answer(prompt, llm, tokenizer)
        print(f"回答: {answer}")

    elif args.mode == "evaluate":
        # 加载数据集
        dataset = load_json_dataset(args.dataset)
        questions = [item["instruction"] for item in dataset]
        references = [item["output"] for item in dataset]

        print(f"总共加载了 {len(questions)} 个问题进行评估")

        # 批量评估
        results = batch_evaluate(questions, references, llm, tokenizer, batch_size=args.batch_size)

        # 保存评估结果
        save_batch_qa_results(results, output_file=args.output)

        # 打印BERTScore
        print("评估结果 (BERTScore):")
        for metric, score in results["bertscore"].items():
            print(f"{metric.capitalize()}: {score:.4f}")

"""
BERTScore:
Precision: 0.6351
Recall: 0.6957
F1: 0.6632
"""
