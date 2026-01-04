"""
ChatLaw Flask API服务
提供RESTful API接口
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from model_training import LegalRAGModel
import faiss
import json
import time

app = Flask(__name__)
CORS(app)  # 允许跨域

# 加载模型
print("🔄 加载ChatLaw模型...")
model = LegalRAGModel(
    embedding_model_path='/workspace/models/ChatLaw-Text2Vec',
    llm_model_path='/workspace/models/Ziya-LLaMA-13B-v1.1'
)


# 加载知识库
model.index = faiss.read_index('legal_knowledge.index')
with open('train_data.jsonl', 'r', encoding='utf-8') as f:
    model.knowledge_base = [json.loads(line) for line in f]

print("✅ 模型加载完成！")

# 简单的Web界面HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatLaw API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-group {
            display: flex;
            gap: 10px;
        }
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-secondary {
            background: #e0e0e0;
            color: #333;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 10px;
            display: none;
        }
        .result.show {
            display: block;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .answer-box {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin-bottom: 20px;
        }
        .reference-box {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 3px solid #ff9800;
            font-size: 14px;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .api-info {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            font-size: 14px;
        }
        .api-info code {
            background: #fff;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚖️ ChatLaw</h1>
        <p class="subtitle">智能法律助手 API Demo</p>
        
        <div class="input-group">
            <textarea id="question" rows="4" placeholder="请输入您的法律问题，例如：如何起诉离婚？"></textarea>
        </div>
        
        <div class="button-group">
            <button class="btn-primary" onclick="askQuestion()">🚀 提问</button>
            <button class="btn-secondary" onclick="clearResult()">🗑️ 清空</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>ChatLaw正在思考中...</p>
        </div>
        
        <div class="result" id="result">
            <h3>💬 ChatLaw 回答：</h3>
            <div class="answer-box" id="answer"></div>
            
            <h4>📚 参考案例：</h4>
            <div id="references"></div>
            
            <p style="color: #999; font-size: 12px; margin-top: 15px;">
                响应时间: <span id="responseTime"></span> 秒
            </p>
        </div>
        
        <div class="api-info">
            <h4>📡 API 使用说明</h4>
            <p><strong>接口地址：</strong> <code>POST /api/ask</code></p>
            <p><strong>请求示例：</strong></p>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
{
  "question": "如何起诉离婚？",
  "top_k": 3
}</pre>
        </div>
    </div>
    
    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('请输入问题！');
                return;
            }
            
            // 显示加载动画
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            
            const startTime = Date.now();
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        question: question,
                        top_k: 3
                    })
                });
                
                const data = await response.json();
                const endTime = Date.now();
                const responseTime = ((endTime - startTime) / 1000).toFixed(2);
                
                if (data.success) {
                    // 显示答案
                    document.getElementById('answer').innerHTML = data.answer;
                    
                    // 显示参考
                    const refsHtml = data.references.map((ref, idx) => `
                        <div class="reference-box">
                            <strong>案例 ${idx + 1} （相似度: ${(ref.score * 100).toFixed(1)}%）</strong><br>
                            <strong>问：</strong>${ref.question}<br>
                            <strong>答：</strong>${ref.answer.substring(0, 200)}...
                        </div>
                    `).join('');
                    document.getElementById('references').innerHTML = refsHtml;
                    
                    document.getElementById('responseTime').textContent = responseTime;
                    
                    // 显示结果
                    document.getElementById('result').classList.add('show');
                } else {
                    alert('错误: ' + data.message);
                }
            } catch (error) {
                alert('请求失败: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }
        
        function clearResult() {
            document.getElementById('question').value = '';
            document.getElementById('result').classList.remove('show');
        }
        
        // 支持Enter键提交
        document.getElementById('question').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

# ===== API路由 =====

@app.route('/')
def index():
    """Web界面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    问答API
    
    请求格式:
    {
        "question": "如何起诉离婚？",
        "top_k": 3
    }
    
    返回格式:
    {
        "success": true,
        "question": "如何起诉离婚？",
        "answer": "...",
        "references": [...]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'message': '缺少question参数'
            }), 400
        
        question = data['question']
        top_k = data.get('top_k', 3)
        
        # 检索
        start_time = time.time()
        retrieved = model.retrieve(question, top_k=top_k)
        
        # 生成答案
        answer = model.generate_answer(question, retrieved)
        
        end_time = time.time()
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'references': retrieved,
            'response_time': round(end_time - start_time, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/retrieve', methods=['POST'])
def retrieve_cases():
    """
    案例检索API
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        results = model.retrieve(query, top_k=top_k)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model': 'ChatLaw RAG',
        'knowledge_base_size': len(model.knowledge_base)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ChatLaw API Server Starting...")
    print("="*60)
    print("📡 访问地址: http://0.0.0.0:5000")
    print("📚 API文档: http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
