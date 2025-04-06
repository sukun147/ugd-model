import random
from collections import defaultdict

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from py2neo import Graph, NodeMatcher
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Neo4jKnowledgeGraph:
    """基于名称查询的Neo4j知识图谱访问类"""

    def __init__(self, uri, user, password):
        """
        初始化Neo4j连接
        Args:
            uri: Neo4j数据库URI (e.g., "bolt://localhost:7687")
            user: 用户名
            password: 密码
        """
        self.graph = Graph(uri, auth=(user, password))
        self.node_matcher = NodeMatcher(self.graph)
        self._build_name_index()

    def _build_name_index(self):
        """构建名称到节点的索引"""
        self.name_to_node = defaultdict(list)
        self.node_to_name = {}

        for node in self.node_matcher.match():
            self.name_to_node[node['name']].append(node)
            self.node_to_name[node.identity] = node['name']

    def get_nodes_by_name(self, name):
        """根据名称获取节点列表"""
        return self.name_to_node.get(name, [])

    def get_node_name(self, node):
        """获取节点的主要名称"""
        return self.node_to_name.get(node.identity, f"Node_{node.identity}")

    def get_neighbors(self, node):
        """获取节点的所有邻居节点(关系类型, 邻居节点)"""
        relationships = list(self.graph.relationships.match((node, None)))
        return [(rel.type, rel.end_node) for rel in relationships]

    def get_relation(self, start_node, end_node):
        """获取两个节点之间的关系类型"""
        rels = list(self.graph.relationships.match((start_node, end_node)))
        return rels[0].type if rels else None

    def get_random_nodes(self, n=1):
        """随机获取n个节点"""
        all_nodes = list(self.node_to_name.keys())
        if not all_nodes:
            return []
        random_ids = random.sample(all_nodes, min(n, len(all_nodes)))
        return [self.graph.nodes[node_id] for node_id in random_ids]


class SentenceTransformerEmbedding:
    """使用Sentence-Transformers生成嵌入"""

    def __init__(self, model_name='intfloat/multilingual-e5-small', device='cpu'):
        """
        Args:
            model_name: Sentence-Transformers模型名称
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, text):
        """为文本生成嵌入"""
        text = "query: " + text
        return self.model.encode(text, convert_to_tensor=False)

    def embed_node(self, node):
        """为Neo4j节点生成嵌入"""
        # 使用节点标签和属性创建文本描述
        labels = list(node.labels)
        properties = dict(node)

        # 排除name/title属性避免重复
        properties_text = ' '.join(f"{k}={v}" for k, v in properties.items()
                                   if k not in ['name', 'title'])

        text = f"{' '.join(labels)}: {properties_text}"
        return self(text)


class PolicyNetwork(nn.Module):
    """策略网络，使用GRPO算法"""

    def __init__(self, state_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, 1)  # 输出单个动作的价值
        self.log_std = nn.Parameter(torch.zeros(1))  # 对数标准差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return dist.Normal(mean, std)


class GRPOKnowledgeExtractor:
    """基于GRPO算法的知识提取器"""

    def __init__(self, kg, embedding_model, context_embedding_dim=384,
                 max_path_length=5, device='cpu', gamma=0.99, clip_param=0.2):
        """
        Args:
            kg: Neo4jKnowledgeGraph实例
            embedding_model: 嵌入模型
            context_embedding_dim: 上下文嵌入维度
            max_path_length: 最大路径长度
            device: 计算设备
            gamma: 折扣因子
            clip_param: PPO裁剪参数
        """
        self.kg = kg
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.embedding_dim
        self.context_embedding_dim = context_embedding_dim
        self.max_path_length = max_path_length
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param

        # 策略网络和价值网络
        self.state_dim = self.embedding_dim * 2
        self.policy_net = PolicyNetwork(self.state_dim).to(device)
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

        # 上下文相关性映射矩阵
        self.W = nn.Linear(self.embedding_dim, context_embedding_dim, bias=False).to(device)

        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters()},
            {'params': self.value_net.parameters()},
            {'params': self.W.parameters()}
        ], lr=3e-4)

        # 经验回放缓冲区
        self.buffer = []
        self.batch_size = 64
        self.ppo_epochs = 4
        self.entropy_coef = 0.01

    def get_entity_embedding(self, node):
        """获取实体嵌入"""
        return self.embedding_model.embed_node(node)

    def get_state_representation(self, current_node, target_node):
        """获取状态表示"""
        e_current = self.get_entity_embedding(current_node)
        e_target = self.get_entity_embedding(target_node)
        return np.concatenate([e_current, e_target - e_current])

    def compute_rewards(self, path, question_embedding):
        """计算路径的奖励"""
        if not path:
            return 0.0, {'reach': 0.0, 'context_related': 0.0, 'conciseness': 0.0, 'total': 0.0}

        # 1. 可达性奖励
        reached_target = 1 if path[-1][2] in self.target_nodes else 0
        r_reach = 1.0 if reached_target else -1.0

        # 2. 上下文相关性奖励
        path_embeddings = []
        for h, r, t in path:
            h_emb = self.get_entity_embedding(h)
            t_emb = self.get_entity_embedding(t)
            path_embeddings.extend([h_emb, t_emb])

        path_embedding = np.mean(path_embeddings, axis=0)

        # 计算余弦相似度
        path_proj = self.W(torch.FloatTensor(path_embedding).to(self.device))
        question_tensor = torch.FloatTensor(question_embedding).to(self.device)
        r_cr = F.cosine_similarity(path_proj.unsqueeze(0),
                                   question_tensor.unsqueeze(0)).item()

        # 3. 简洁性奖励
        r_cs = 1.0 / len(path)

        # 总奖励 (可调整权重)
        total_reward = r_reach + r_cr + r_cs

        return total_reward, {
            'reach': r_reach,
            'context_related': r_cr,
            'conciseness': r_cs,
            'total': total_reward
        }

    def extract_path(self, source_node, target_node_names, question_embedding,
                     epsilon=0.1, verbose=False):
        """提取知识路径"""
        # 将目标名称转换为节点对象
        self.target_nodes = []
        for name in target_node_names:
            self.target_nodes.extend(self.kg.get_nodes_by_name(name))

        if not self.target_nodes:
            if verbose:
                print("No target nodes found for names:", target_node_names)
            return [], {'reach': -1.0, 'context_related': 0.0, 'conciseness': 0.0, 'total': -1.0}

        current_node = source_node
        path = []
        visited = set()
        log_probs = []
        values = []
        states = []
        actions = []

        for step in range(self.max_path_length):
            visited.add(current_node.identity)

            # 随机选择一个目标节点来计算状态表示
            target_node = random.choice(self.target_nodes)
            state = self.get_state_representation(current_node, target_node)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 获取状态值
            with torch.no_grad():
                value = self.value_net(state_tensor).squeeze().item()
                action_dist = self.policy_net(state_tensor)

            # 获取可能的动作(邻居节点)
            neighbors = self.kg.get_neighbors(current_node)
            neighbor_nodes = [t for r, t in neighbors if t.identity not in visited]  # 避免循环

            if not neighbor_nodes:
                if verbose:
                    print(f"Step {step}: No unvisited neighbors")
                break

            # 准备邻居节点的状态表示
            neighbor_states = []
            for neighbor in neighbor_nodes:
                neighbor_state = self.get_state_representation(neighbor, target_node)
                neighbor_states.append(neighbor_state)

            neighbor_states_tensor = torch.FloatTensor(np.array(neighbor_states)).to(self.device)

            # 选择动作
            if random.random() < epsilon:
                action_idx = random.randint(0, len(neighbor_nodes) - 1)
                if verbose:
                    print(f"Step {step}: Random exploration")
            else:
                # 使用策略网络选择动作
                with torch.no_grad():
                    action_dist = self.policy_net(neighbor_states_tensor)
                    action_values = action_dist.mean.squeeze()
                    action_idx = torch.argmax(action_values).item()

                if verbose:
                    print(f"Step {step}: Greedy action (value={action_values[action_idx].item():.2f})")

            next_node = neighbor_nodes[action_idx]
            relation = self.kg.get_relation(current_node, next_node)

            if relation is None:  # 确保关系存在
                if verbose:
                    print(
                        f"No relation between {self.kg.get_node_name(current_node)} and {self.kg.get_node_name(next_node)}")
                break

            # 记录动作概率和状态值
            action_state = self.get_state_representation(next_node, target_node)
            action_state_tensor = torch.FloatTensor(action_state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_dist = self.policy_net(action_state_tensor)
                log_prob = action_dist.log_prob(action_dist.mean).item()

            log_probs.append(log_prob)
            values.append(value)
            states.append(state)
            actions.append(action_idx)

            # 记录路径
            path.append((current_node, relation, next_node))

            # 检查是否到达目标
            if next_node in self.target_nodes:
                if verbose:
                    print(f"Reached target node {self.kg.get_node_name(next_node)}")
                break

            current_node = next_node

        # 计算奖励
        total_reward, reward_components = self.compute_rewards(path, question_embedding)

        # 存储经验
        if path:
            discounted_rewards = []
            R = 0
            for r in reversed([total_reward] * len(path)):
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)

            self.buffer.append({
                'states': states,
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
                'rewards': discounted_rewards,
                'path': path
            })

        return path, reward_components

    def update(self):
        """使用GRPO算法更新策略"""
        if len(self.buffer) < self.batch_size:
            return

        # 准备批量数据
        batch = random.sample(self.buffer, self.batch_size)

        # 合并所有经验
        all_states = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_returns = []

        for experience in batch:
            all_states.extend(experience['states'])
            all_actions.extend(experience['actions'])
            all_log_probs.extend(experience['log_probs'])
            all_values.extend(experience['values'])
            all_returns.extend(experience['rewards'])

        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions_tensor = torch.LongTensor(all_actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        old_values_tensor = torch.FloatTensor(all_values).to(self.device)

        # 计算优势
        advantages = returns_tensor - old_values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            # 获取新策略的动作概率
            dist = self.policy_net(states_tensor)
            new_log_probs = dist.log_prob(dist.mean).gather(1, actions_tensor.unsqueeze(1)).squeeze()

            # 计算概率比
            ratio = (new_log_probs - old_log_probs_tensor).exp()

            # 计算裁剪的PPO目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            new_values = self.value_net(states_tensor).squeeze()
            value_loss = F.mse_loss(new_values, returns_tensor)

            # 熵奖励
            entropy = dist.entropy().mean()

            # 总损失
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            # 梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.W.parameters(), 0.5)
            self.optimizer.step()

        # 清空缓冲区
        self.buffer = []

    def train(self, questions, num_epochs=10, initial_epsilon=0.2):
        """训练知识提取器"""
        epsilon = initial_epsilon

        for epoch in range(num_epochs):
            epoch_rewards = []

            for question in tqdm(questions, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                source_node_names = question['source_entities']
                target_node_names = question['target_entities']
                question_embedding = question['context_embedding']

                # 获取源节点
                source_nodes = []
                for name in source_node_names:
                    source_nodes.extend(self.kg.get_nodes_by_name(name))

                if not source_nodes:
                    continue

                for source_node in source_nodes:
                    # 提取路径并收集经验
                    path, rewards = self.extract_path(
                        source_node, target_node_names, question_embedding, epsilon
                    )
                    epoch_rewards.append(rewards['total'])

                    # 更新策略
                    if len(self.buffer) >= self.batch_size:
                        self.update()

            # 打印统计信息
            if epoch_rewards:
                avg_reward = np.mean(epoch_rewards)
                print(f"Epoch {epoch + 1}: Avg Reward = {avg_reward:.4f}, Epsilon = {epsilon:.3f}")
            else:
                print(f"Epoch {epoch + 1}: No valid training data")

            # 衰减探索率
            epsilon = max(0.05, epsilon * 0.95)  # 保持最小探索率

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'W_state_dict': self.W.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.W.load_state_dict(checkpoint['W_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def prepare_training_data(kg, embedding_model, num_questions=100):
    """准备训练数据(基于名称)"""
    questions = []

    for _ in range(num_questions):
        # 随机选择源节点和目标节点
        source_node = kg.get_random_nodes(1)
        target_node = kg.get_random_nodes(1)

        if not source_node or not target_node:
            continue

        source_node = source_node[0]
        target_node = target_node[0]

        # 获取节点名称
        source_name = kg.get_node_name(source_node)
        target_name = kg.get_node_name(target_node)

        # 创建问题文本
        question_text = f"What is the relationship between {source_name} and {target_name}?"

        # 生成问题嵌入
        question_embedding = embedding_model(question_text)

        # 创建问题
        question = {
            'source_entities': [source_name],
            'target_entities': [target_name],
            'context_embedding': question_embedding,
            'text': question_text
        }
        questions.append(question)

    return questions


def main():
    # 1. 初始化Neo4j连接
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "sukun031015"

    print("Connecting to Neo4j...")
    try:
        kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        print(f"Connected to Neo4j. Found {len(kg.name_to_node)} unique names.")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    # 2. 初始化嵌入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading Sentence Transformer model...")
    try:
        embedding_model = SentenceTransformerEmbedding(
            model_name='intfloat/multilingual-e5-small',
            device=device
        )
        print(f"Loaded model with embedding dimension: {embedding_model.embedding_dim}")
    except Exception as e:
        print(f"Failed to load embedding model: {e}")
        return

    # 3. 创建GRPO知识提取器
    grpo_extractor = GRPOKnowledgeExtractor(
        kg=kg,
        embedding_model=embedding_model,
        context_embedding_dim=embedding_model.embedding_dim,  # 使用模型的实际维度
        max_path_length=5,
        device=device,
        gamma=0.99,
        clip_param=0.2
    )

    # 4. 准备训练数据
    print("Preparing training data...")
    try:
        train_questions = prepare_training_data(kg, embedding_model, num_questions=200)
        print(f"Prepared {len(train_questions)} training questions.")

        if not train_questions:
            print("No training questions prepared. Check your Neo4j data.")
            return
    except Exception as e:
        print(f"Failed to prepare training data: {e}")
        return

    # 5. 训练模型
    print("Starting training...")
    try:
        grpo_extractor.train(train_questions, num_epochs=10, initial_epsilon=0.2)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # 6. 示例推理
    print("\nRunning example inference...")
    try:
        source_node = kg.get_random_nodes(1)
        target_node = kg.get_random_nodes(1)

        if not source_node or not target_node:
            print("Not enough nodes for example inference")
            return

        source_node = source_node[0]
        target_node = target_node[0]

        source_name = kg.get_node_name(source_node)
        target_name = kg.get_node_name(target_node)

        print(f"Source node: {source_name}")
        print(f"Target node: {target_name}")

        question_text = f"What is the relationship between {source_name} and {target_name}?"
        question_embedding = embedding_model(question_text)

        path, rewards = grpo_extractor.extract_path(
            source_node,
            [target_name],
            question_embedding,
            verbose=True
        )

        print("\nExtracted path:")
        for h, r, t in path:
            h_name = kg.get_node_name(h)
            t_name = kg.get_node_name(t)
            print(f"{h_name} --[{r}]--> {t_name}")

        print("\nRewards:", rewards)
    except Exception as e:
        print(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
