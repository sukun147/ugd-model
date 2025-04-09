from transformers import AutoTokenizer

from ner.ner import ner_with_pretrained_model
from extractor.extractor import KnowledgeExtractor
from vllm import LLM, SamplingParams

MODEL_PATH = "D:/Code/model/Qwen2.5-7B-MedChatZH-LoRA-SFT-GPTQ-Int4"


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


if __name__ == "__main__":
    question = "生姜有什么功能？"
    answer, knowledge = qa(question)
    print(answer)
    print(knowledge)
