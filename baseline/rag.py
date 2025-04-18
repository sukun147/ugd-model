import argparse
import gc
import json
import time

import evaluate
import torch
from py2neo import Graph
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


class KnowledgeGraphLoader:
    def __init__(self, uri, user, password):
        self.graph = Graph(uri, auth=(user, password))

    def load_knowledge_graph(self):
        query = """
        MATCH (h)-[r]->(t)
        RETURN h.name AS head, t.name AS tail, r.type AS relation
        """
        # 执行Cypher查询并获取结果
        result = self.graph.run(query)
        triples = [(record["head"], record["tail"], record["relation"]) for record in result]
        return triples


# 2. Vectorize the knowledge base
def vectorize_knowledge_base(sentences, model, tokenizer):
    formatted_sentences = ["query: " + sentence for sentence in sentences]

    # 处理可能的大量三元组，分批次处理
    batch_size = 128
    all_embeddings = []

    for i in range(0, len(formatted_sentences), batch_size):
        batch = formatted_sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings)

        # 及时清理内存
        del inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 合并所有嵌入
    if len(all_embeddings) == 1:
        return all_embeddings[0]
    else:
        return torch.cat(all_embeddings, dim=0)


# 3. Vectorize the input query
def vectorize_query(query, model, tokenizer):
    formatted_query = "query: " + query
    inputs = tokenizer(formatted_query, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return query_embedding


# 批量向量化查询
def batch_vectorize_queries(queries, model, tokenizer, batch_size=16):
    all_embeddings = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        formatted_queries = ["query: " + query for query in batch_queries]
        inputs = tokenizer(formatted_queries, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeddings)

        # 及时清理内存
        del inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return torch.cat(all_embeddings, dim=0)


# 4. Top-k retrieval
def retrieve_top_k(query_embedding, knowledge_embeddings, sentences, k=5):
    scores = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)[0]
    top_k_indices = torch.topk(scores, k=k).indices
    return [sentences[i] for i in top_k_indices]


# 批量检索
def batch_retrieve_top_k(query_embeddings, knowledge_embeddings, sentences, k=5):
    # 分批计算相似度，避免OOM
    batch_size = 32
    all_retrieved_knowledge = []

    for i in range(0, len(query_embeddings), batch_size):
        batch_query_embeddings = query_embeddings[i:i + batch_size]

        # 计算相似度
        scores = util.pytorch_cos_sim(batch_query_embeddings, knowledge_embeddings)
        top_k_indices = torch.topk(scores, k=k, dim=1).indices

        # 收集检索到的知识
        for j in range(len(batch_query_embeddings)):
            query_knowledge = [sentences[idx] for idx in top_k_indices[j]]
            all_retrieved_knowledge.append(query_knowledge)

        # 清理内存
        del scores, top_k_indices
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_retrieved_knowledge


# 5. Generate answer using retrieved knowledge
def generate_answer(prompt, knowledge, model, tokenizer):
    input_text = prompt + "\n" + "\n".join(knowledge)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=16384)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 清理内存
    del inputs, outputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return answer


# 批量生成答案
def batch_generate_answers(prompt, batch_knowledge, model, tokenizer, batch_size=5):
    all_answers = []

    for i in range(0, len(batch_knowledge), batch_size):
        current_batch = batch_knowledge[i:i + batch_size]
        batch_inputs = []

        for knowledge in current_batch:
            input_text = prompt + "\n" + "\n".join(knowledge)
            encoded = tokenizer(input_text, truncation=True, max_length=16384, return_tensors="pt")
            batch_inputs.append(encoded)

        batch_answers = []
        for inputs in batch_inputs:
            outputs = model.generate(**inputs, max_length=16384)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            batch_answers.append(answer)

            # 清理内存
            del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        all_answers.extend(batch_answers)

        # 强制清理内存
        del batch_inputs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_answers


# 6. Load JSON dataset
def load_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return data


# 7. Evaluation metrics - 使用 evaluate 库替代 datasets.load_metric
def compute_metrics_batch(predictions, references):
    """分批计算评估指标，防止OOM"""
    print("开始分批计算评估指标...")

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 加载评估指标
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    # 分批计算BLEU
    print("计算BLEU分数...")
    batch_size = 200
    bleu_results = []
    for i in range(0, len(predictions), batch_size):
        end_idx = min(i + batch_size, len(predictions))
        print(f"BLEU: 处理 {i + 1} 到 {end_idx} / {len(predictions)}")

        batch_preds = predictions[i:end_idx]
        batch_refs = references[i:end_idx]

        # 为每个引用创建一个列表
        batch_refs_list = [[ref] for ref in batch_refs]

        result = bleu.compute(predictions=batch_preds, references=batch_refs_list)
        bleu_results.append(result)

        # 清理内存
        gc.collect()

    # 合并BLEU结果
    bleu_score = sum(res['bleu'] for res in bleu_results) / len(bleu_results) if bleu_results else 0

    # 分批计算ROUGE
    print("计算ROUGE分数...")
    rouge_results = []
    for i in range(0, len(predictions), batch_size):
        end_idx = min(i + batch_size, len(predictions))
        print(f"ROUGE: 处理 {i + 1} 到 {end_idx} / {len(predictions)}")

        batch_preds = predictions[i:end_idx]
        batch_refs = references[i:end_idx]

        result = rouge.compute(predictions=batch_preds, references=batch_refs)
        rouge_results.append(result)

        # 清理内存
        gc.collect()

    # 合并ROUGE结果
    rouge_score = {}
    if rouge_results:
        # 初始化合并结果字典
        for key in rouge_results[0].keys():
            rouge_score[key] = sum(res[key] for res in rouge_results) / len(rouge_results)

    # 分批计算BERTScore
    print("计算BERTScore分数...")
    batch_size = 32  # BERTScore需要更小的批次
    all_P, all_R, all_F1 = [], [], []

    for i in range(0, len(predictions), batch_size):
        end_idx = min(i + batch_size, len(predictions))
        print(f"BERTScore: 处理 {i + 1} 到 {end_idx} / {len(predictions)}")

        batch_preds = predictions[i:end_idx]
        batch_refs = references[i:end_idx]

        try:
            result = bertscore.compute(
                predictions=batch_preds,
                references=batch_refs,
                lang="zh"
            )
            all_P.extend(result["precision"])
            all_R.extend(result["recall"])
            all_F1.extend(result["f1"])
        except Exception as e:
            print(f"BERTScore计算异常: {e}")
            # 如果出现异常，填充零值
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

    return {"bleu": bleu_score}, rouge_score, bertscore_score


# 批量处理问答
def batch_qa(questions, embedding_model, embedding_tokenizer, gen_model, gen_tokenizer,
             knowledge_embeddings, sentences, prompt, batch_size=5):
    """批量处理问题，显示简单进度信息"""
    results = []
    total_batches = (len(questions) + batch_size - 1) // batch_size  # 向上取整计算批次数

    print(f"开始处理 {len(questions)} 个问题，共 {total_batches} 个批次")
    start_time = time.time()

    # 批量向量化所有查询
    print("正在向量化所有查询...")
    all_query_embeddings = batch_vectorize_queries(questions, embedding_model, embedding_tokenizer, batch_size=32)

    # 批量检索知识
    print("正在为所有查询检索知识...")
    all_retrieved_knowledge = batch_retrieve_top_k(all_query_embeddings, knowledge_embeddings, sentences, k=5)

    # 释放不再需要的向量化资源
    del all_query_embeddings
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 批量生成答案
    print("正在生成所有答案...")
    for i in range(0, len(questions), batch_size):
        batch_number = i // batch_size + 1
        print(f"正在处理批次 {batch_number}/{total_batches}...")

        batch_questions = questions[i:i + batch_size]
        batch_knowledge = all_retrieved_knowledge[i:i + batch_size]

        # 生成答案
        batch_answers = batch_generate_answers(prompt, batch_knowledge, gen_model, gen_tokenizer, batch_size=batch_size)

        # 保存结果
        for j, (question, answer, knowledge) in enumerate(zip(batch_questions, batch_answers, batch_knowledge)):
            results.append({
                "question": question,
                "answer": answer,
                "knowledge": knowledge
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

        # 及时保存中间结果
        if batch_number % 5 == 0 or batch_number == total_batches:
            temp_file = f"rag_results_temp_{batch_number}of{total_batches}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"中间结果已保存至 {temp_file}")

    total_time = time.time() - start_time
    print(f"批处理完成，共处理 {len(questions)} 个问题，总用时: {total_time:.1f}秒，"
          f"平均每个问题: {total_time / len(questions):.1f}秒")

    return results


# 从批处理结果中评估性能
def evaluate_from_batch_results(batch_results, references):
    predictions = [result["answer"] for result in batch_results]
    bleu, rouge, bertscore = compute_metrics_batch(predictions, references)
    return {
        "BLEU": bleu,
        "ROUGE": rouge,
        "BERTScore": bertscore
    }


# 保存批处理结果到文件
def save_batch_results(batch_results, output_file="rag_batch_results.json"):
    """保存批处理结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至 {output_file}")


# 释放模型资源
def release_model_resources(embedding_model, embedding_tokenizer, gen_model, gen_tokenizer):
    """释放模型资源以释放内存"""
    del embedding_model
    del embedding_tokenizer
    del gen_model
    del gen_tokenizer
    # 强制清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# 评估功能
def evaluate_dataset(dataset_path, embedding_model, embedding_tokenizer, gen_model, gen_tokenizer,
                     knowledge_embeddings, sentences, prompt, batch_size=5):
    # 尝试从文件加载预处理结果
    result_file = "rag_evaluation_results.json"
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
        print(f"开始批量处理，批次大小为 {batch_size}")
        batch_results = batch_qa(questions, embedding_model, embedding_tokenizer, gen_model, gen_tokenizer,
                                 knowledge_embeddings, sentences, prompt, batch_size)

        # 保存结果
        save_batch_results(batch_results, result_file)

    # 释放模型资源以便评估
    print("释放模型资源以便评估...")
    release_model_resources(embedding_model, embedding_tokenizer, gen_model, gen_tokenizer)

    # 计算评估指标
    print("计算评估指标...")
    references = [item["output"] for item in load_json_dataset(dataset_path)]
    metrics = evaluate_from_batch_results(batch_results, references)

    return metrics, batch_results


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG (Retrieval-Augmented Generation) 系统")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "batch", "evaluate"],
                        help="运行模式: single(单个问题), batch(批量问答), evaluate(评估)")
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

    args = parser.parse_args()

    # 加载知识图谱
    print("正在从Neo4j加载知识图谱...")
    # 使用py2neo连接数据库
    loader = KnowledgeGraphLoader(uri="bolt://localhost:7687", user="neo4j", password="sukun031015")
    triples = loader.load_knowledge_graph()
    sentences = ["头实体{}和尾实体{}的关系是{}".format(h, t, r) for h, t, r in triples]
    print(f"成功加载 {len(triples)} 条三元组")

    # 加载嵌入模型
    print("正在加载嵌入模型...")
    embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    embedding_model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    # 加载生成模型
    print("正在加载生成模型...")
    gen_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen2.5-7B-MedChatZH-LoRA-SFT")
    gen_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/models/Qwen2.5-7B-MedChatZH-LoRA-SFT")

    # 向量化知识库
    print("正在向量化知识库...")
    knowledge_embeddings = vectorize_knowledge_base(sentences, embedding_model, embedding_tokenizer)
    print("知识库向量化完成")

    # 提示词
    prompt = "你作为中医诊疗专家，请基于下列检索获得的中医知识进行回答，确保专业性与可读性的平衡，最终形成逻辑缜密、重点突出的中医知识回答："

    if args.mode == "single":
        # 单个问题测试
        question = args.question
        print(f"问题: {question}")

        # 向量化查询
        query_embedding = vectorize_query(question, embedding_model, embedding_tokenizer)

        # 检索知识
        top_k_knowledge = retrieve_top_k(query_embedding, knowledge_embeddings, sentences)
        print("检索到的知识:")
        for i, k in enumerate(top_k_knowledge):
            print(f"{i + 1}. {k}")

        # 生成答案
        answer = generate_answer(prompt, top_k_knowledge, gen_model, gen_tokenizer)
        print(f"回答: {answer}")

    elif args.mode == "batch":
        # 批量问答测试
        if args.questions:
            questions = args.questions.split(";")
            print(f"接收到 {len(questions)} 个问题，开始批量处理...")
        else:
            print("未提供问题列表，使用默认问题进行测试...")
            questions = ["生姜有什么功能？", "胃痛怎么办？", "中暑有什么症状？", "附子的功效是什么？", "如何治疗感冒？"]

        batch_results = batch_qa(questions, embedding_model, embedding_tokenizer, gen_model, gen_tokenizer,
                                 knowledge_embeddings, sentences, prompt, args.batch_size)
        save_batch_results(batch_results, args.output)

        # 打印结果摘要
        for i, result in enumerate(batch_results):
            print(f"问题 {i + 1}: {result['question']}")
            print(f"回答: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"回答: {result['answer']}")
            print("-" * 50)

    elif args.mode == "evaluate":
        # 评估测试集
        print(f"开始评估测试集: {args.dataset}")

        metrics, _ = evaluate_dataset(args.dataset, embedding_model, embedding_tokenizer, gen_model, gen_tokenizer,
                                      knowledge_embeddings, sentences, prompt, args.batch_size)

        print("\n评估结果:")
        print("BLEU:", metrics["BLEU"])
        print("ROUGE:", metrics["ROUGE"])
        print("BERTScore:")
        for metric, score in metrics["BERTScore"].items():
            print(f"  {metric}: {score}")
