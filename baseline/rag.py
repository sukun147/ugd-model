import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_metric
from sentence_transformers import util
from neo4j import GraphDatabase

class KnowledgeGraphLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_knowledge_graph(self):
        query = """
        MATCH (h)-[r]->(t)
        RETURN h.name AS head, t.name AS tail, r.type AS relation
        """
        with self.driver.session() as session:
            result = session.run(query)
            triples = [(record["head"], record["tail"], record["relation"]) for record in result]
        return triples


# 2. Vectorize the knowledge base
def vectorize_knowledge_base(sentences, model, tokenizer):
    formatted_sentences = ["query: " + sentence for sentence in sentences]
    inputs = tokenizer(formatted_sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


# 3. Vectorize the input query
def vectorize_query(query, model, tokenizer):
    formatted_query = "query: " + query
    inputs = tokenizer(formatted_query, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return query_embedding


# 4. Top-k retrieval
def retrieve_top_k(query_embedding, knowledge_embeddings, sentences, k=5):
    scores = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)[0]
    top_k_indices = torch.topk(scores, k=k).indices
    return [sentences[i] for i in top_k_indices]


# 5. Generate answer using retrieved knowledge
def generate_answer(prompt, knowledge, model, tokenizer):
    input_text = prompt + "\n" + "\n".join(knowledge)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 6. Load JSON dataset
def load_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return data


# 7. Evaluation metrics
def compute_metrics(predictions, references):
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    bertscore = load_metric("bertscore")

    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="zh")

    return bleu_score, rouge_score, bertscore_score


# Main function
if __name__ == "__main__":
    # Load knowledge graph from Neo4j
    loader = KnowledgeGraphLoader(uri="bolt://localhost:7687", user="neo4j", password="sukun031015")
    triples = loader.load_knowledge_graph()
    loader.close()
    sentences = ["头实体{}和尾实体{}的关系是{}".format(h, t, r) for h, t, r in triples]

    # Load embedding model
    embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    embedding_model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    # Load Qwen2.5-7B-Instruct model
    gen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    gen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # Vectorize knowledge base
    knowledge_embeddings = vectorize_knowledge_base(sentences, embedding_model, embedding_tokenizer)

    # Load JSON dataset
    dataset = load_json_dataset("../llm/data/MedChatZH_valid.json")
    predictions, references = [], []
    prompt = "你作为中医诊疗专家，请基于下列检索获得的中医知识进行回答，确保专业性与可读性的平衡，最终形成逻辑缜密、重点突出的中医知识阐释。"
    for item in dataset:
        question = item["instruction"]
        reference = item["output"]
        query_embedding = vectorize_query(question, embedding_model, embedding_tokenizer)
        top_k_knowledge = retrieve_top_k(query_embedding, knowledge_embeddings, sentences)
        prediction = generate_answer(prompt, top_k_knowledge, gen_model, gen_tokenizer)
        predictions.append(prediction)
        references.append(reference)

    # Compute metrics
    bleu, rouge, bertscore = compute_metrics(predictions, references)
    print("BLEU:", bleu)
    print("ROUGE:", rouge)
    print("BERTScore:", bertscore)
