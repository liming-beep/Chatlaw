"""
修复Ziya-LLaMA乱码问题的补丁
问题：tokenizer和生成参数配置不正确
"""

import os

# 修复后的模型加载代码
FIXED_MODEL_LOADING = '''
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
'''

FIXED_GENERATE = '''
    def generate_answer(self, query, retrieved_docs, max_length=512):
        """生成答案（修复版）"""
        if not self.llm_model or not self.tokenizer:
            return "模型未加载"
        
        try:
            # 构建上下文
            context = "\\n\\n".join([
                f"案例{i+1}：\\n问：{doc['question']}\\n答：{doc['answer']}"
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
'''

print("=" * 60)
print("🔧 Ziya-LLaMA 乱码修复工具")
print("=" * 60)

# 读取app_streamlit.py
app_file = 'app_streamlit.py'
if not os.path.exists(app_file):
    print(f"❌ 未找到 {app_file}")
    print("请在包含app_streamlit.py的目录下运行此脚本")
    exit(1)

with open(app_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 备份
backup_file = app_file + '.backup_ziya'
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"✅ 已备份到: {backup_file}")

# 检查是否需要修复
if 'use_fast=False' in content and 'max_new_tokens' in content:
    print("✅ 文件已经包含修复，无需重复修复")
else:
    print("\n⚠️  检测到需要手动修复")
    print("\n请按照以下步骤操作：")
    print("\n" + "=" * 60)
    print("📝 手动修复步骤")
    print("=" * 60)
    
    print("\n1️⃣ 找到 load_llm 方法（约第50-100行），修改tokenizer加载：")
    print("""
在这一行：
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.llm_model_path,
        trust_remote_code=True
    )

改成：
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.llm_model_path,
        trust_remote_code=True,
        use_fast=False,      # ← 添加这行
        padding_side='left'  # ← 添加这行
    )
""")
    
    print("\n2️⃣ 找到 generate_answer 方法（约第150-200行），修改prompt格式：")
    print("""
找到构建prompt的部分，改成Ziya的格式：

prompt = f\"\"\"<human>:以下是一些法律案例作为参考：

{context}

现在请根据以上案例，回答这个问题：{query}

请提供专业、准确的法律建议。
<bot>:\"\"\"
""")
    
    print("\n3️⃣ 修改生成参数（同样在 generate_answer 方法中）：")
    print("""
找到 model.generate() 调用，修改参数：

outputs = self.llm_model.generate(
    **inputs,
    max_new_tokens=512,        # ← 改用max_new_tokens
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    pad_token_id=self.tokenizer.pad_token_id,
    eos_token_id=self.tokenizer.eos_token_id,
)
""")
    
    print("\n4️⃣ 修改解码方式（generate_answer方法末尾）：")
    print("""
找到 tokenizer.decode()，改成：

generated_text = self.tokenizer.decode(
    outputs[0][inputs['input_ids'].shape[1]:],  # ← 只解码新生成的部分
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
""")

print("\n" + "=" * 60)
print("💾 完整修复版代码已生成")
print("=" * 60)

# 生成完整的修复版方法
with open('fixed_methods.py', 'w', encoding='utf-8') as f:
    f.write("""# 修复后的完整方法代码
# 复制这些方法替换app_streamlit.py中对应的方法

import torch

""")
    f.write(FIXED_MODEL_LOADING)
    f.write("\n\n")
    f.write(FIXED_GENERATE)

print("✅ 已生成修复代码: fixed_methods.py")
print("   你可以参考这个文件中的代码进行修改")

print("\n" + "=" * 60)
print("🚀 修复完成后重启应用")
print("=" * 60)
print("streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501")