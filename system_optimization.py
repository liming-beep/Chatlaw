"""
ChatLaw系统优化方案
选择优化方向：模型性能优化
"""

# ===============================================
# 优化方案1: 模型推理加速（推荐⭐⭐⭐⭐⭐）
# ===============================================

"""
针对T4 GPU的性能优化策略：

1. 模型量化
   - 使用8bit量化减少显存占用（已实现）
   - 可进一步使用4bit量化（需要bitsandbytes库）
   
2. 批量推理
   - 实现问题批处理，提高吞吐量
   
3. 缓存机制
   - 对高频问题进行缓存
   - 使用Redis存储查询结果
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from functools import lru_cache
import hashlib

class OptimizedLegalRAG:
    """优化版本的RAG模型"""
    
    def __init__(self, model_path, use_4bit=False):
        """
        Args:
            use_4bit: 是否使用4bit量化（更省显存，略微降低精度）
        """
        if use_4bit:
            # 4bit量化配置（需要 pip install bitsandbytes）
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            # 8bit量化
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map="auto"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 缓存配置
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, question):
        """生成缓存键"""
        return hashlib.md5(question.encode()).hexdigest()
    
    def generate_with_cache(self, question, context):
        """带缓存的生成"""
        cache_key = self._get_cache_key(question)
        
        # 检查缓存
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # 生成答案
        self.cache_misses += 1
        answer = self._generate(question, context)
        
        # 存入缓存
        self.cache[cache_key] = answer
        return answer
    
    def _generate(self, question, context):
        """实际生成函数"""
        prompt = f"上下文：{context}\n\n问题：{question}\n\n答案："
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                num_beams=2,  # 使用beam search提高质量
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.split("答案：")[-1].strip()
    
    def get_cache_stats(self):
        """获取缓存统计"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate * 100:.2f}%"
        }


# ===============================================
# 优化方案2: 检索质量优化
# ===============================================

class ImprovedRetrieval:
    """改进的检索模块"""
    
    def __init__(self, embedder, knowledge_base):
        self.embedder = embedder
        self.knowledge_base = knowledge_base
        
        # 使用更高级的索引结构
        import faiss
        
        # 生成向量
        questions = [item['question'] for item in knowledge_base]
        embeddings = embedder.encode(questions, show_progress_bar=True)
        
        # 使用HNSW索引（更快的近似最近邻搜索）
        dimension = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32是连接数
        
        # 归一化并添加
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def retrieve_with_rerank(self, query, top_k=3, rerank_top_n=10):
        """
        两阶段检索：粗排+精排
        
        1. 粗排：快速检索top-N候选
        2. 精排：使用更精确的模型重排top-k
        """
        # 第一阶段：粗排
        query_vec = self.embedder.encode([query])
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(
            query_vec.astype('float32'), 
            rerank_top_n
        )
        
        candidates = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.knowledge_base):
                candidates.append({
                    'item': self.knowledge_base[idx],
                    'coarse_score': float(score)
                })
        
        # 第二阶段：精排（使用更详细的相似度计算）
        # 这里可以使用cross-encoder等更精确的模型
        # 简化版本：基于问题+答案的综合相似度
        for candidate in candidates:
            full_text = candidate['item']['question'] + " " + candidate['item']['answer']
            full_vec = self.embedder.encode([full_text])
            faiss.normalize_L2(full_vec)
            
            # 计算精确相似度
            refined_score = np.dot(query_vec[0], full_vec[0])
            candidate['refined_score'] = float(refined_score)
        
        # 按精排分数排序
        candidates.sort(key=lambda x: x['refined_score'], reverse=True)
        
        # 返回top-k
        results = []
        for candidate in candidates[:top_k]:
            results.append({
                'question': candidate['item']['question'],
                'answer': candidate['item']['answer'],
                'score': candidate['refined_score']
            })
        
        return results


# ===============================================
# 优化方案3: 并发处理优化
# ===============================================

from concurrent.futures import ThreadPoolExecutor
import asyncio

class ConcurrentRAG:
    """支持并发处理的RAG系统"""
    
    def __init__(self, model, max_workers=4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def batch_answer(self, questions):
        """批量处理问题"""
        futures = []
        for q in questions:
            future = self.executor.submit(self.model.answer_question, q)
            futures.append(future)
        
        # 等待所有任务完成
        results = [future.result() for future in futures]
        return results
    
    async def async_answer(self, question):
        """异步问答"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self.model.answer_question, 
            question
        )
        return result


# ===============================================
# 性能基准测试
# ===============================================

class PerformanceBenchmark:
    """性能测试工具"""
    
    @staticmethod
    def benchmark_inference_speed(model, test_questions, num_runs=10):
        """测试推理速度"""
        import time
        
        times = []
        for _ in range(num_runs):
            for q in test_questions:
                start = time.time()
                _ = model.answer_question(q)
                end = time.time()
                times.append(end - start)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'qps': 1 / np.mean(times)  # 每秒查询数
        }
    
    @staticmethod
    def benchmark_memory_usage():
        """测试显存使用"""
        if torch.cuda.is_available():
            return {
                'allocated': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                'cached': f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                'max_allocated': f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            }
        return None
    
    @staticmethod
    def plot_optimization_results(baseline, optimized):
        """绘制优化对比图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 速度对比
        categories = ['基线模型', '优化模型']
        times = [baseline['avg_time'], optimized['avg_time']]
        
        axes[0].bar(categories, times, color=['#ff6b6b', '#4ecdc4'])
        axes[0].set_ylabel('平均响应时间 (秒)')
        axes[0].set_title('推理速度对比')
        
        for i, v in enumerate(times):
            axes[0].text(i, v + 0.1, f'{v:.2f}s', ha='center')
        
        # 加速比
        speedup = baseline['avg_time'] / optimized['avg_time']
        axes[1].bar(['加速比'], [speedup], color='#95e1d3')
        axes[1].set_ylabel('倍数')
        axes[1].set_title(f'性能提升: {speedup:.2f}x')
        axes[1].text(0, speedup + 0.1, f'{speedup:.2f}x', ha='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('optimization_results.png', dpi=300)
        print("✅ 优化对比图已保存: optimization_results.png")


# ===============================================
# 优化效果展示
# ===============================================

if __name__ == "__main__":
    print("="*60)
    print("⚡ ChatLaw 系统优化演示")
    print("="*60)
    
    # 优化前后对比
    test_questions = [
        "如何起诉离婚？",
        "劳动合同纠纷怎么处理？",
        "房屋买卖合同注意事项"
    ]
    
    print("\n📊 优化策略：")
    print("1. ✅ 模型量化（8bit → 4bit）")
    print("2. ✅ 查询缓存机制")
    print("3. ✅ 两阶段检索（粗排+精排）")
    print("4. ✅ 并发处理优化")
    
    print("\n📈 预期优化效果：")
    print("- 推理速度提升: 2-3x")
    print("- 显存占用降低: 30-40%")
    print("- 检索准确率提升: 5-10%")
    print("- 并发吞吐量提升: 3-4x")
    
    print("\n" + "="*60)
