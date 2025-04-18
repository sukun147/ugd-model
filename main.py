import json
import time

from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from extractor.extractor import KnowledgeExtractor
from ner.ner import ner_with_pretrained_model

MODEL_PATH = "D:/Code/model/Qwen2.5-7B-MedChatZH-LoRA-SFT"
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


def evaluate_responses_with_batch():
    """使用批处理方式评估回答"""
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

    # 确保资源已初始化（在批量处理前只初始化一次）
    initialize_resources()

    # 批量处理问题
    batch_size = 5  # 可以根据您的GPU内存调整批量大小
    print(f"开始批量处理，批次大小为 {batch_size}")

    batch_results = batch_qa(questions, batch_size)
    predictions = [result["answer"] for result in batch_results]

    print("计算评估指标...")
    # 计算 BLEU 分数
    bleu_scores = []
    smooth_fn = SmoothingFunction().method1
    for ref, pred in zip(references, predictions):
        bleu_score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
        bleu_scores.append(bleu_score)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # 计算 ROUGE 分数
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    # 计算 BERTScore 分数
    P, R, F1 = bert_score(predictions, references, lang="zh", verbose=True)
    avg_bertscore = {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item()
    }

    # 结果输出
    return {
        "BLEU": avg_bleu,
        "ROUGE": rouge_scores,
        "BERTScore": avg_bertscore,
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
    parser.add_argument("--batch_size", type=int, default=5,
                        help="批处理大小")
    parser.add_argument("--output", type=str, default="batch_qa_results.json",
                        help="批量问答结果输出文件")

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
        print("BLEU 分数:", results["BLEU"])
        print("ROUGE 分数:")
        for metric, score in results["ROUGE"].items():
            print(f"  {metric}: {score}")
        print("BERTScore:")
        for metric, score in results["BERTScore"].items():
            print(f"  {metric}: {score}")
