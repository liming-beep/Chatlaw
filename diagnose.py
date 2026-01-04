"""
ChatLaw 环境诊断工具
检查模型、数据、依赖是否正常
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("🔍 ChatLaw 环境诊断")
print("=" * 60)

# 1. 检查Python环境
print("\n1️⃣ Python环境")
print(f"   Python版本: {sys.version}")
print(f"   当前目录: {os.getcwd()}")

# 2. 检查依赖包
print("\n2️⃣ 依赖包检查")
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'sentence_transformers': 'Sentence Transformers',
    'faiss': 'Faiss',
    'streamlit': 'Streamlit',
    'sklearn': 'Scikit-learn'
}

for package, name in packages.items():
    try:
        __import__(package)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - 未安装")

# 3. 检查GPU
print("\n3️⃣ GPU检查")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"   ⚠️  CUDA不可用，将使用CPU")
except Exception as e:
    print(f"   ❌ 检查失败: {e}")

# 4. 检查模型文件
print("\n4️⃣ 模型文件检查")

# 可能的模型路径
model_paths = [
    "/workspace/models/ChatLaw-Text2Vec",
    "/workspace/models/Ziya-LLaMA-13B-v1.1"
]

found_models = []
for path in model_paths:
    if os.path.exists(path):
        size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
        size_mb = size / 1024 / 1024
        print(f"   ✅ 找到: {path} ({size_mb:.0f} MB)")
        found_models.append(path)

if not found_models:
    print("   ❌ 未找到模型文件")
    print("\n   💡 建议：")
    print("   1. 检查模型是否下载完整")
    print("   2. 确认模型路径是否正确")
    print("   3. 运行: find /workspace -name '*chatlaw*' -type d")

# 5. 检查数据文件
print("\n5️⃣ 数据文件检查")

data_files = [
    'train_data.jsonl',
    'val_data.jsonl',
    'test_data.jsonl',
    'legal_qa_dataset.jsonl',
    'data/train_data.jsonl',
    'data/legal_qa_dataset.jsonl',
    '/workspace/data/train_data.jsonl',
    '/workspace/data/legal_qa_dataset.jsonl',
]

found_data = []
for path in data_files:
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            size = os.path.getsize(path) / 1024
            print(f"   ✅ 找到: {path} ({lines} 条, {size:.0f} KB)")
            found_data.append(path)
        except:
            print(f"   ⚠️  文件存在但无法读取: {path}")

if not found_data:
    print("   ❌ 未找到数据文件")
    print("\n   💡 建议：")
    print("   1. 运行: python quick_data_generator.py")
    print("   2. 运行: python process_legal_data.py")

# 6. 检查知识库索引
print("\n6️⃣ 知识库索引检查")

index_files = [
    'legal_knowledge.index',
    'data/legal_knowledge.index',
    '/workspace/data/legal_knowledge.index',
]

found_index = False
for path in index_files:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"   ✅ 找到: {path} ({size:.0f} KB)")
        found_index = True
        break

if not found_index:
    print("   ⚠️  未找到索引文件")
    print("   💡 这是正常的，首次运行会自动构建")

# 7. 检查配置文件
print("\n7️⃣ Streamlit配置检查")

config_path = os.path.expanduser("~/.streamlit/config.toml")
if os.path.exists(config_path):
    print(f"   ✅ 配置文件存在: {config_path}")
    with open(config_path, 'r') as f:
        content = f.read()
        if '0.0.0.0' in content:
            print(f"   ✅ 配置正确（使用 0.0.0.0）")
        else:
            print(f"   ⚠️  配置可能不正确")
else:
    print(f"   ⚠️  配置文件不存在")
    print(f"   💡 建议创建配置文件")

# 8. 总结建议
print("\n" + "=" * 60)
print("📋 诊断总结")
print("=" * 60)

issues = []
solutions = []

if not found_models:
    issues.append("❌ 未找到模型文件")
    solutions.append("方案A: 使用演示版（不需要模型）: streamlit run app_demo.py")
    solutions.append("方案B: 确认模型下载路径，修改app_streamlit.py中的模型路径")

if not found_data:
    issues.append("❌ 未找到数据文件")
    solutions.append("运行: python quick_data_generator.py 生成数据")

if issues:
    print("\n⚠️  发现问题：")
    for issue in issues:
        print(f"   {issue}")
    
    print("\n💡 解决方案：")
    for i, sol in enumerate(solutions, 1):
        print(f"   {i}. {sol}")
else:
    print("\n✅ 环境检查通过！")
    print("   可以运行: streamlit run app_streamlit.py")

print("\n" + "=" * 60)
print("诊断完成！")
print("=" * 60)

# 9. 查找模型的辅助命令
print("\n🔍 辅助命令（可选）：")
print("   查找chatlaw模型: find /workspace -name '*chatlaw*' -type d 2>/dev/null")
print("   查找ziya模型: find /workspace -name '*ziya*' -type d 2>/dev/null")
print("   查看当前目录: ls -lh")
print("   查看data目录: ls -lh data/")
