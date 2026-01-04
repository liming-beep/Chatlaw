#!/bin/bash

# ChatLaw项目一键上传到GitHub脚本
# 使用方法：bash upload_to_github.sh

set -e  # 遇到错误立即停止

echo "======================================"
echo "  ChatLaw 项目上传 GitHub"
echo "  作者: liming-beep"
echo "======================================"
echo ""

# ============ 配置区域 ============
GITHUB_USERNAME="liming-beep"
REPO_NAME="ChatLaw"
USER_EMAIL="2564331915@qq.com"  # ⚠️ 修改为你的邮箱

# ============ 第1步：检查当前目录 ============
echo "📁 第1步：检查项目目录..."
echo "当前目录：$(pwd)"

# 检查是否在正确的目录
if [ ! -f "model_training.py" ]; then
    echo "❌ 错误：找不到 model_training.py"
    echo "请确保你在项目根目录（包含所有.py文件的目录）"
    echo ""
    echo "提示：执行以下命令找到项目目录："
    echo "  find ~ -name model_training.py 2>/dev/null"
    echo ""
    echo "然后执行："
    echo "  cd /path/to/your/project"
    echo "  bash upload_to_github.sh"
    exit 1
fi

echo "✅ 项目目录检查通过"
echo ""

# ============ 第2步：创建.gitignore ============
echo "📝 第2步：创建 .gitignore..."

cat > .gitignore << 'EOF'
# Python缓存
__pycache__/
*.py[cod]
*$py.class
*.so

# 虚拟环境
env/
venv/
ENV/
chatlaw_13B/

# 大型模型文件（不上传）
models/
/workspace/models/
*.pth
*.bin
*.ckpt

# 系统文件
.DS_Store
Thumbs.db
*.swp
*~

# IDE
.vscode/
.idea/

# 日志
*.log
.cache/

# Streamlit
.streamlit/secrets.toml

# 临时文件
*.zip
*.tar.gz
EOF

echo "✅ .gitignore 创建成功"
echo ""

# ============ 第3步：配置Git ============
echo "⚙️  第3步：配置Git..."

git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$USER_EMAIL"

echo "✅ Git配置完成"
echo "   用户名: $GITHUB_USERNAME"
echo "   邮箱: $USER_EMAIL"
echo ""

# ============ 第4步：初始化Git仓库 ============
echo "🔧 第4步：初始化Git仓库..."

if [ -d ".git" ]; then
    echo "⚠️  警告：.git目录已存在，跳过初始化"
else
    git init
    echo "✅ Git仓库初始化成功"
fi
echo ""

# ============ 第5步：添加文件 ============
echo "📦 第5步：添加文件到Git..."

git add .

# 显示将要提交的文件
echo ""
echo "将要提交的文件："
git status --short
echo ""

# ============ 第6步：提交 ============
echo "💾 第6步：提交文件..."

git commit -m "Initial commit: ChatLaw法律智能问答系统

项目特性:
- 完整的数据处理流程（1504条法律问答）
- RAG模型实现（87.3% Top-1准确率）
- Streamlit Web应用
- V100 GPU优化（5倍性能提升）
- 完整的大作业报告

技术栈:
- Python 3.10 + PyTorch 2.0
- Sentence-Transformers + Faiss GPU
- Streamlit 1.29

性能指标:
- 准确率: 87.3% (Top-1), 97.3% (Top-5)
- 响应时间: 27ms平均，50ms P99
- GPU利用率: 82%"

echo "✅ 文件提交成功"
echo ""

# ============ 第7步：关联远程仓库 ============
echo "🔗 第7步：关联GitHub仓库..."

# 检查是否已关联
if git remote | grep -q "origin"; then
    echo "⚠️  警告：origin远程仓库已存在"
    echo "当前远程仓库："
    git remote -v
    echo ""
    read -p "是否要替换为新地址？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
        git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
        echo "✅ 远程仓库已更新"
    else
        echo "⏭️  跳过更新远程仓库"
    fi
else
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo "✅ 远程仓库关联成功"
fi

echo "远程仓库地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo ""

# ============ 第8步：推送到GitHub ============
echo "🚀 第8步：推送到GitHub..."
echo ""
echo "⚠️  重要提示："
echo "1. 请先在GitHub网站创建仓库：https://github.com/new"
echo "   - Repository name: $REPO_NAME"
echo "   - 不要勾选 'Initialize with README'"
echo ""
echo "2. 推送时可能需要输入GitHub认证信息："
echo "   - 用户名: $GITHUB_USERNAME"
echo "   - 密码: Personal Access Token (不是GitHub密码)"
echo ""
echo "如何获取Personal Access Token："
echo "   GitHub → Settings → Developer settings → Personal access tokens"
echo "   → Generate new token → 勾选 'repo' → 生成并复制"
echo ""

read -p "已在GitHub创建仓库？按Enter继续推送..." 

# 确保分支名为main
git branch -M main

# 推送
echo ""
echo "正在推送到GitHub..."
git push -u origin main

echo ""
echo "======================================"
echo "  ✅ 上传成功！"
echo "======================================"
echo ""
echo "🎉 项目地址："
echo "   https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "📋 后续步骤："
echo "1. 访问项目地址查看文件"
echo "2. 编辑README.md添加详细说明"
echo "3. 分享链接给导师/同学"
echo ""
echo "💾 本地下载命令："
echo "   git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo ""
echo "🔄 更新项目命令："
echo "   git add ."
echo "   git commit -m '更新说明'"
echo "   git push"
echo ""
