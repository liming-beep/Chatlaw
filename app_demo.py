"""
ChatLaw 简化演示版
- 只做检索，不做生成
- 使用在线模型，无需本地模型
- 快速部署，适合演示
"""

import streamlit as st
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# 页面配置
st.set_page_config(
    page_title="ChatLaw 法律智能助手",
    page_icon="⚖️",
    layout="wide"
)

# CSS样式
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196F3;
    }
    .bot-message {
        background: linear-gradient(135deg, #f1f8e9, #dcedc8);
        border-left: 5px solid #4CAF50;
    }
    .reference-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #ff9800;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_knowledge_base():
    """加载知识库"""
    try:
        # 尝试多个路径
        for path in ['train_data.jsonl', 'legal_qa_dataset.jsonl', 
                     'data/train_data.jsonl', 'data/legal_qa_dataset.jsonl',
                     '/workspace/data/train_data.jsonl', '/workspace/data/legal_qa_dataset.jsonl']:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                    if len(data) > 0:
                        return data, path
            except FileNotFoundError:
                continue
        
        # 如果都找不到，返回示例数据
        st.warning("⚠️ 未找到数据文件，使用示例数据")
        return get_demo_data(), "demo"
        
    except Exception as e:
        st.error(f"加载失败: {e}")
        return get_demo_data(), "demo"


def get_demo_data():
    """示例数据"""
    return [
        {
            'question': '如何起诉离婚？',
            'answer': '起诉离婚的基本流程：\n\n1. **准备材料**\n   - 起诉状（一式三份）\n   - 身份证明（身份证、户口本）\n   - 结婚证\n   - 感情破裂证据\n\n2. **提交诉讼**\n   - 向被告户籍所在地或经常居住地法院提交\n   - 法院受理后缴纳诉讼费\n\n3. **等待开庭**\n   - 一般1-2周收到传票\n   - 准备出庭应诉\n\n4. **判决执行**\n   - 第一次起诉通常不判离\n   - 6个月后可再次起诉\n\n**法律依据**：《民法典》第1079条\n**注意**：建议咨询专业律师',
            'category': '婚姻法'
        },
        {
            'question': '劳动合同纠纷怎么处理？',
            'answer': '劳动合同纠纷处理途径：\n\n1. **协商解决**（首选）\n   - 与用人单位直接沟通\n   - 成本最低，时间最快\n\n2. **申请调解**\n   - 企业劳动争议调解委员会\n   - 基层人民调解组织\n\n3. **劳动仲裁**（前置程序）\n   - 必须先仲裁，再诉讼\n   - 自知道权利被侵害之日起1年内申请\n   - 免费，45天内结案\n\n4. **法院诉讼**\n   - 对仲裁不服可起诉\n   - 15天内提起\n\n**所需材料**：\n- 劳动合同\n- 工资条、考勤记录\n- 社保缴纳证明\n\n**法律依据**：《劳动争议调解仲裁法》',
            'category': '劳动法'
        },
        {
            'question': '合同违约如何赔偿？',
            'answer': '合同违约责任承担方式：\n\n1. **继续履行**\n   - 对方要求继续履行合同义务\n\n2. **采取补救措施**\n   - 修理、重作、更换等\n\n3. **赔偿损失**\n   - 实际损失 + 可得利益损失\n   - 需提供损失证据\n\n4. **支付违约金**\n   - 按合同约定\n   - 过高可请求调低（超损失30%）\n   - 过低可请求增加\n\n5. **定金罚则**\n   - 给付方违约：定金不退\n   - 收受方违约：双倍返还\n\n**注意**：\n- 违约金和定金不能同时主张\n- 可同时主张违约金和赔偿损失\n\n**法律依据**：《民法典》第577、585条',
            'category': '合同法'
        },
        {
            'question': '交通事故怎么处理？',
            'answer': '交通事故处理流程：\n\n1. **现场处理**\n   - 立即停车、开警示灯\n   - 设置警告标志\n   - 保护现场、拍照取证\n   - 轻微事故可快速处理\n\n2. **报警报保险**\n   - 严重事故必须报警\n   - 48小时内报保险\n\n3. **责任认定**\n   - 全责、主责、同责、次责、无责\n   - 不服可申请复核\n\n4. **赔偿协商**\n   - 医疗费、误工费、护理费\n   - 交通费、营养费\n   - 财产损失\n\n5. **保险理赔**\n   - 交强险：12.2万\n   - 商业险：按保额\n\n**注意**：\n- 保留所有票据\n- 及时就医并固定证据\n\n**法律依据**：《道路交通安全法》',
            'category': '交通法'
        },
        {
            'question': '工伤如何认定和赔偿？',
            'answer': '工伤认定与赔偿：\n\n**一、工伤认定情形**\n1. 工作时间、工作场所内，因工作原因受伤\n2. 上下班途中，非本人主要责任的交通事故\n3. 工作时间前后，预备/收尾工作受伤\n4. 职业病\n5. 因工外出期间受伤\n\n**二、申请流程**\n1. 30天内向社保部门申请认定\n2. 认定后申请劳动能力鉴定\n3. 根据伤残等级计算赔偿\n\n**三、赔偿项目**\n- 医疗费、护理费、住院伙食补助\n- 停工留薪期工资（原工资）\n- 一次性伤残补助金\n- 一次性工伤医疗补助金\n- 一次性伤残就业补助金\n\n**四、赔偿标准**\n- 1-4级：保留劳动关系，月发伤残津贴\n- 5-6级：保留劳动关系，难安排工作月发津贴\n- 7-10级：解除合同时支付一次性补助\n\n**法律依据**：《工伤保险条例》',
            'category': '劳动法'
        }
    ]


class SimpleLegalRetriever:
    """简单的法律检索器（基于TF-IDF）"""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.questions = [item['question'] for item in knowledge_base]
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
    
    def search(self, query, top_k=3):
        """检索最相关的案例"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.question_vectors)[0]
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相似度的结果
                results.append({
                    'question': self.kb[idx]['question'],
                    'answer': self.kb[idx]['answer'],
                    'category': self.kb[idx]['category'],
                    'score': float(similarities[idx])
                })
        
        return results


# 侧边栏
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #667eea;'>⚖️ ChatLaw</h1>
            <p style='color: #666;'>法律智能助手（演示版）</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 系统状态
    st.subheader("📊 系统状态")
    
    # 加载知识库
    with st.spinner('加载知识库...'):
        knowledge_base, data_path = load_knowledge_base()
        retriever = SimpleLegalRetriever(knowledge_base)
    
    st.success(f"✅ 知识库已加载")
    st.info(f"📁 {len(knowledge_base)} 条法律问答")
    if data_path != "demo":
        st.caption(f"数据源: {data_path}")
    else:
        st.caption("数据源: 演示数据")
    
    st.markdown("---")
    
    # 设置
    st.subheader("⚙️ 检索设置")
    top_k = st.slider("检索案例数量", 1, 10, 3)
    
    st.markdown("---")
    
    st.info("""
    **💡 使用说明**
    1. 在右侧输入法律问题
    2. 系统自动检索相关案例
    3. 展示最匹配的法律建议
    
    **🎯 适用场景**
    - 婚姻法律咨询
    - 劳动纠纷处理
    - 合同法律问题
    - 交通事故处理
    - 工伤认定赔偿
    
    **⚠️ 免责声明**
    本系统仅供参考，不构成专业法律意见。具体问题请咨询专业律师。
    """)


# 主界面
st.title("💬 ChatLaw 法律智能问答")
st.markdown("请输入您的法律问题，我将为您检索相关的法律建议")

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
        
        if 'references' in message:
            with st.expander("📚 查看参考案例", expanded=False):
                for i, ref in enumerate(message['references']):
                    st.markdown(f"""
                        <div class="reference-box">
                            <b>📄 案例 {i+1} - {ref['category']} (相似度: {ref['score']:.1%})</b><br>
                            <b>问：</b>{ref['question']}<br>
                            <b>答：</b>{ref['answer'][:150]}...
                        </div>
                    """, unsafe_allow_html=True)

# 输入区域
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "您的问题：",
        placeholder="例如：如何起诉离婚？",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    submit_button = st.button("🚀 提问", key="submit", type="primary")

# 快捷问题
st.markdown("**💡 快捷问题：**")
quick_qs = ["如何起诉离婚？", "劳动合同纠纷怎么处理？", "交通事故怎么处理？", "工伤如何认定？"]
cols = st.columns(4)
for idx, qq in enumerate(quick_qs):
    if cols[idx].button(qq, key=f"quick_{idx}"):
        user_input = qq
        submit_button = True

# 处理提问
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
    
    # 检索答案
    with st.spinner('🔍 正在检索相关案例...'):
        start_time = time.time()
        results = retriever.search(user_input, top_k=top_k)
        end_time = time.time()
    
    if results:
        # 使用最佳匹配的答案
        best_answer = results[0]['answer']
        
        # 添加助手消息
        st.session_state.messages.append({
            'role': 'assistant',
            'content': best_answer,
            'references': results
        })
        
        # 显示答案
        st.markdown(f"""
            <div class="chat-message bot-message">
                <b>🤖 ChatLaw：</b><br>
                {best_answer}
            </div>
        """, unsafe_allow_html=True)
        
        # 显示参考案例
        with st.expander("📚 查看所有参考案例", expanded=True):
            for i, ref in enumerate(results):
                st.markdown(f"""
                    <div class="reference-box">
                        <b>📄 案例 {i+1} - {ref['category']} (相似度: {ref['score']:.1%})</b><br>
                        <b>问：</b>{ref['question']}<br>
                        <b>答：</b>{ref['answer']}
                    </div>
                """, unsafe_allow_html=True)
        
        st.success(f"✅ 检索完成！耗时: {end_time - start_time:.2f}秒")
    else:
        st.error("❌ 未找到相关案例，请尝试换个问法")
    
    # 刷新页面
    st.rerun()

# 页脚
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>⚖️ ChatLaw 法律智能助手（演示版） | 基于TF-IDF检索 | 数据来源: ChatLaw知识库</p>
        <p style='font-size: 0.8rem;'>⚠️ 本系统仅供学习演示，实际法律问题请咨询专业律师</p>
    </div>
""", unsafe_allow_html=True)