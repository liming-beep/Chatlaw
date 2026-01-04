"""
ChatLaw Web应用部署
使用Streamlit构建交互式界面
"""

import streamlit as st
import torch
import json
from model_training import LegalRAGModel
import time

# 页面配置
st.set_page_config(
    page_title="ChatLaw 法律智能助手",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 5px solid #4CAF50;
    }
    .reference-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.3rem;
        border-left: 3px solid #ff9800;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# 初始化模型（使用缓存避免重复加载）
@st.cache_resource
def load_model():
    """加载模型（只加载一次）"""
    with st.spinner('🔄 正在加载ChatLaw模型，请稍候...'):
        model = LegalRAGModel(
            embedding_model_path="/workspace/models/ChatLaw-Text2Vec",
            llm_model_path="/workspace/models/Ziya-LLaMA-13B-v1.1"
        )
        # 加载知识库索引
        import faiss
        model.index = faiss.read_index('legal_knowledge.index')
        with open('train_data.jsonl', 'r', encoding='utf-8') as f:
            model.knowledge_base = [json.loads(line) for line in f]
    return model

# 侧边栏
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/4CAF50/white?text=ChatLaw", 
             use_container_width=True)
    st.title("⚖️ ChatLaw")
    st.markdown("---")
    
    # 功能选择
    mode = st.radio(
        "选择功能模式",
        ["💬 智能问答", "📊 案例检索", "📈 系统统计"],
        index=0
    )
    
    st.markdown("---")
    
    # 设置选项
    st.subheader("⚙️ 参数设置")
    top_k = st.slider("检索相关案例数", 1, 10, 3)
    max_length = st.slider("答案最大长度", 128, 1024, 512)
    temperature = st.slider("生成温度", 0.1, 1.0, 0.7, 0.1)
    
    st.markdown("---")
    st.info("""
    **使用说明：**
    1. 输入法律相关问题
    2. 系统自动检索相关案例
    3. 生成专业法律建议
    
    **适用场景：**
    - 婚姻法咨询
    - 劳动纠纷
    - 合同问题
    - 继承分配
    """)

# 主界面
if mode == "💬 智能问答":
    st.title("💬 ChatLaw 法律智能问答")
    st.markdown("请输入您的法律问题，我将为您提供专业的法律建议")
    
    # 初始化会话状态
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 显示历史对话
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>👤 您的问题：</b><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <b>🤖 ChatLaw：</b><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
            
            # 显示参考案例
            if 'references' in message:
                with st.expander("📚 查看参考案例"):
                    for i, ref in enumerate(message['references']):
                        st.markdown(f"""
                            <div class="reference-box">
                                <b>案例 {i+1}（相似度: {ref['score']:.2%}）</b><br>
                                <b>问：</b>{ref['question']}<br>
                                <b>答：</b>{ref['answer'][:200]}...
                            </div>
                        """, unsafe_allow_html=True)
    
    # 输入框
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "请输入您的问题：",
            placeholder="例如：如何起诉离婚？",
            key="user_input",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.button("🚀 发送", use_container_width=True)
    
    # 快捷问题
    st.markdown("**💡 快捷问题：**")
    quick_questions = [
        "如何起诉离婚？",
        "劳动合同纠纷怎么处理？",
        "房屋买卖合同注意事项",
        "遗产继承顺序是什么？"
    ]
    cols = st.columns(4)
    for idx, qq in enumerate(quick_questions):
        if cols[idx].button(qq, key=f"quick_{idx}"):
            user_input = qq
            submit_button = True
    
    # 处理问答
    if submit_button and user_input:
        # 添加用户消息
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # 显示用户问题
        st.markdown(f"""
            <div class="chat-message user-message">
                <b>👤 您的问题：</b><br>
                {user_input}
            </div>
        """, unsafe_allow_html=True)
        
        # 生成回答
        with st.spinner('🤔 ChatLaw正在思考...'):
            try:
                model = load_model()
                
                # 检索
                retrieved = model.retrieve(user_input, top_k=top_k)
                
                # 生成答案
                answer = model.generate_answer(user_input, retrieved)
                
                # 添加助手消息
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'references': retrieved
                })
                
                # 显示答案
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <b>🤖 ChatLaw：</b><br>
                        {answer}
                    </div>
                """, unsafe_allow_html=True)
                
                # 显示参考
                with st.expander("📚 查看参考案例"):
                    for i, ref in enumerate(retrieved):
                        st.markdown(f"""
                            <div class="reference-box">
                                <b>案例 {i+1}（相似度: {ref['score']:.2%}）</b><br>
                                <b>问：</b>{ref['question']}<br>
                                <b>答：</b>{ref['answer'][:200]}...
                            </div>
                        """, unsafe_allow_html=True)
                
                st.success("✅ 回答完成！")
                
            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")
        
        # 清空输入
        st.rerun()

elif mode == "📊 案例检索":
    st.title("📊 法律案例检索系统")
    
    query = st.text_input("输入关键词检索相关案例：")
    
    if st.button("🔍 检索") and query:
        model = load_model()
        results = model.retrieve(query, top_k=10)
        
        st.success(f"找到 {len(results)} 个相关案例")
        
        for i, result in enumerate(results):
            with st.expander(f"案例 {i+1} - 相似度: {result['score']:.2%}"):
                st.markdown(f"**问题：** {result['question']}")
                st.markdown(f"**答案：** {result['answer']}")

elif mode == "📈 系统统计":
    st.title("📈 ChatLaw 系统统计")
    
    # 知识库统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("知识库案例数", "1,234", "+56")
    with col2:
        st.metric("累计咨询次数", "5,678", "+234")
    with col3:
        st.metric("平均响应时间", "2.3秒", "-0.5秒")
    
    # 图表
    st.subheader("📊 使用趋势")
    import pandas as pd
    import numpy as np
    
    # 模拟数据
    dates = pd.date_range('2025-01-01', periods=30)
    data = pd.DataFrame({
        '日期': dates,
        '咨询量': np.random.randint(50, 200, 30)
    })
    
    st.line_chart(data.set_index('日期'))

# 页脚
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>⚖️ ChatLaw - AI法律助手 | 基于RAG技术 | Powered by ChatLaw-Text2Vec & Ziya-LLaMA</p>
        <p style="font-size: 0.8rem;">⚠️ 本系统仅供学习参考，实际法律问题请咨询专业律师</p>
    </div>
""", unsafe_allow_html=True)
