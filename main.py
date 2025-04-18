import json

from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from extractor.extractor import KnowledgeExtractor
from ner.ner import ner_with_pretrained_model

MODEL_PATH = "D:/Code/model/Qwen2.5-7B-MedChatZH-LoRA-SFT-GPTQ-Int4"
DATA_PATH = "llm/data/MedChatZH_valid.json"


def qa(question):
    entities = []
    ner_res = ner_with_pretrained_model([question])
    for ner_token in ner_res[0]:
        entities.append(ner_token.word)
    extractor = KnowledgeExtractor(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="sukun031015",
        embedding_model_name="intfloat/multilingual-e5-small"
    )
    knowledge = extractor.extract_subgraph(entities, [], 1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = f"""
    你是一个中医领域的知识图谱问答助手，你的任务是根据问题和知识图谱中的信息来回答问题。
    问题: {question}
    知识图谱中的信息: {knowledge}
    请根据知识图谱中的信息回答问题。
    """
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    llm = LLM(model=MODEL_PATH, quantization="gptq")
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


def evaluate_responses_line_by_line():
    references = []
    predictions = []

    # 逐行读取数据集
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data["instruction"]
                reference = data["output"]
                references.append(reference)
                answer, _ = qa(question)
                predictions.append(answer)
            except json.JSONDecodeError:
                print(f"跳过无效行: {line.strip()}")
                continue

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


if __name__ == "__main__":
    # question = "生姜有什么功能？"
    # answer, knowledge = qa(question)
    # print(answer)
    # print(knowledge)
    results = evaluate_responses_line_by_line()
    print("评估结果:")
    print("BLEU 分数:", results["BLEU"])
    print("ROUGE 分数:", results["ROUGE"])
