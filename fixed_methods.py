# 修复后的完整方法代码
# 复制这些方法替换app_streamlit.py中对应的方法

import torch


    def load_llm(self):
        """加载生成模型（修复版）"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"📥 加载生成模型...")
            print(f"   路径: {self.llm_model_path}")
            
            # 正确加载Ziya的tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_path,
                trust_remote_code=True,
                use_fast=False,  # ← 关键！Ziya需要use_fast=False
                padding_side='left',  # ← Ziya需要左填充
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_path,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.float16,  # 使用fp16节省显存
                low_cpu_mem_usage=True,
            )
            
            self.llm_model.eval()
            print(f"✅ 生成模型加载成功")
            
        except Exception as e:
            print(f"❌ 生成模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise



    def generate_answer(self, query, retrieved_docs, max_length=512):
        """生成答案（修复版）"""
        if not self.llm_model or not self.tokenizer:
            return "模型未加载"
        
        try:
            # 构建上下文
            context = "\n\n".join([
                f"案例{i+1}：\n问：{doc['question']}\n答：{doc['answer']}"
                for i, doc in enumerate(retrieved_docs[:3])
            ])
            
            # Ziya模型的正确prompt格式
            prompt = f"""<human>:以下是一些法律案例作为参考：

{context}

现在请根据以上案例，回答这个问题：{query}

请提供专业、准确的法律建议。
<bot>:"""
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # 生成（使用正确的参数）
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=max_length,  # ← 用max_new_tokens而不是max_length
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
            
            # 解码（只取生成的部分，不包括prompt）
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],  # ← 关键！只解码新生成的token
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return f"生成出错: {str(e)}"
