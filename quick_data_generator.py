"""
快速生成1000+条高质量法律问答数据
不依赖外部下载，本地直接生成
运行时间：约30秒
"""

import json
import random

# 法律知识库（基于真实法律条文）
LEGAL_KNOWLEDGE_BASE = {
    '婚姻法': {
        '离婚': {
            'questions': [
                '如何起诉离婚？', '离婚需要什么条件？', '离婚流程是什么？',
                '起诉离婚要多久？', '离婚需要准备什么材料？', '协议离婚和诉讼离婚的区别？',
                '离婚可以在异地办理吗？', '分居多久可以自动离婚？', '第一次起诉离婚会判离吗？'
            ],
            'answer_template': '''关于{topic}：

1. 法律依据：《民法典》第1076-1080条

2. 具体情况：
   {details}

3. 注意事项：
   - 保存好相关证据材料
   - 注意诉讼时效
   - 可以委托律师代理

建议：如情况复杂，建议咨询专业律师。''',
            'details': [
                '协议离婚：双方自愿，需到民政局办理，有30天冷静期',
                '诉讼离婚：一方不同意，需向法院提起诉讼，准备感情破裂证据',
                '需提供：身份证、结婚证、户口本、财产证明等材料',
                '第一次起诉通常不判离，需等6个月后再次起诉'
            ]
        },
        '财产分割': {
            'questions': [
                '离婚财产如何分割？', '婚前财产离婚时需要分割吗？', '房产如何分割？',
                '一方出轨财产如何分割？', '父母赠予的财产算共同财产吗？', '股票基金如何分割？'
            ],
            'answer_template': '''关于离婚{topic}：

1. 基本原则：
   - 夫妻共同财产平均分割
   - 照顾子女和女方权益
   - 照顾无过错方

2. 财产类型：
   {details}

3. 特殊情况：
   - 一方隐藏财产：少分或不分
   - 家暴、出轨：无过错方可多分

法律依据：《民法典》第1062、1063、1087条''',
            'details': [
                '婚前财产：归个人所有，不参与分割',
                '婚后财产：工资、奖金、投资收益等属共同财产',
                '房产：需看产权登记、出资情况、还贷情况',
                '债务：共同债务共同承担，个人债务个人承担'
            ]
        },
        '子女抚养': {
            'questions': [
                '离婚后孩子归谁？', '抚养费标准是多少？', '2岁以下孩子归谁？',
                '可以不给抚养费吗？', '抚养权可以变更吗？', '探视权怎么行使？'
            ],
            'answer_template': '''关于子女{topic}：

1. 抚养权原则：
   - 2岁以下一般随母亲
   - 2-8岁考虑现状和双方条件
   - 8岁以上需征求孩子意见

2. 抚养费标准：
   {details}

3. 探视权：
   - 不直接抚养方有探视权
   - 具体时间协商或法院判决

法律依据：《民法典》第1084、1085条''',
            'details': [
                '有固定收入：月收入的20%-30%',
                '无固定收入：参照当地平均收入',
                '包括：生活费、教育费、医疗费',
                '支付到18岁，特殊情况可延长'
            ]
        }
    },
    
    '劳动法': {
        '劳动合同': {
            'questions': [
                '没签劳动合同怎么办？', '试用期有多长？', '公司不签合同违法吗？',
                '劳动合同到期公司不续签？', '试用期可以随时辞职吗？', '合同期限怎么选？'
            ],
            'answer_template': '''关于{topic}：

1. 法律规定：
   {details}

2. 权利保障：
   - 入职1个月内必须签订书面合同
   - 超过1个月未签，支付双倍工资
   - 超过1年视为无固定期限合同

3. 维权途径：
   - 向劳动监察投诉
   - 申请劳动仲裁

法律依据：《劳动合同法》第7、10、82条''',
            'details': [
                '试用期：合同3个月-1年试用期≤2个月，1-3年≤3个月，3年以上≤6个月',
                '必备条款：工作内容、地点、时间、报酬、保险、合同期限',
                '未签合同：第2个月起至1年，每月双倍工资',
                '合同到期：提前30天通知，不续签需支付经济补偿'
            ]
        },
        '工伤赔偿': {
            'questions': [
                '工伤认定标准是什么？', '工伤赔偿包括哪些？', '工伤认定流程？',
                '上下班路上出车祸算工伤吗？', '工伤期间工资怎么算？', '公司不认工伤怎么办？'
            ],
            'answer_template': '''关于{topic}：

1. 工伤认定：
   {details}

2. 赔偿项目：
   - 医疗费、护理费、住院伙食补助
   - 停工留薪期工资（原工资福利不变）
   - 一次性伤残补助金
   - 一次性工伤医疗、就业补助金

3. 申请流程：
   - 30天内向社保部门申请认定
   - 认定后申请劳动能力鉴定
   - 根据伤残等级计算赔偿

法律依据：《工伤保险条例》''',
            'details': [
                '工作时间、工作场所内，因工作原因受伤',
                '上下班途中，非本人主要责任的交通事故',
                '工作时间前后，与工作有关的预备或收尾工作',
                '职业病、因工外出期间受伤、抢险救灾受伤'
            ]
        },
        '辞退补偿': {
            'questions': [
                '被辞退有补偿吗？', '经济补偿金怎么算？', '公司随便辞退员工吗？',
                '违法辞退赔偿多少？', '辞职有补偿吗？', 'N+1补偿是什么？'
            ],
            'answer_template': '''关于{topic}：

1. 经济补偿：
   {details}

2. 赔偿情况：
   - 违法解除：补偿金的2倍
   - 协商解除：支付补偿金
   - 过错解除：无需补偿

3. 特殊规定：
   - 代通知金（N+1）：提前30天通知或额外1个月工资
   - 封顶：月工资3倍社平工资，最多12年

法律依据：《劳动合同法》第46、47、87条''',
            'details': [
                '计算标准：每满1年支付1个月工资',
                '工作6个月以上不满1年：按1年算',
                '不满6个月：支付半个月工资',
                '月工资：解除前12个月平均工资'
            ]
        }
    },
    
    '合同法': {
        '合同效力': {
            'questions': [
                '口头协议有效吗？', '合同什么时候生效？', '没有签字合同有效吗？',
                '合同可以撤销吗？', '什么情况合同无效？', '合同盖章后多久生效？'
            ],
            'answer_template': '''关于{topic}：

1. 有效要件：
   {details}

2. 无效情形：
   - 一方欺诈、胁迫
   - 恶意串通损害他人利益
   - 损害公共利益
   - 违反法律强制性规定

3. 可撤销情形：
   - 重大误解
   - 显失公平
   - 欺诈、胁迫、乘人之危

法律依据：《民法典》合同编''',
            'details': [
                '主体适格：有民事行为能力',
                '意思表示真实：自愿、非欺诈胁迫',
                '内容合法：不违反法律法规',
                '形式合法：法律要求书面的需书面'
            ]
        },
        '违约责任': {
            'questions': [
                '合同违约怎么赔偿？', '违约金最高多少？', '定金和违约金能同时要吗？',
                '不可抗力能免责吗？', '对方违约我能解除合同吗？', '违约金太高能减少吗？'
            ],
            'answer_template': '''关于{topic}：

1. 违约责任：
   {details}

2. 违约金调整：
   - 过高（超过损失30%）：可请求调低
   - 过低：可请求增加
   - 举证责任：主张调整方承担

3. 免责事由：
   - 不可抗力（地震、战争等）
   - 合同约定的免责条款

法律依据：《民法典》第577、585条''',
            'details': [
                '继续履行：对方可要求继续履行合同',
                '赔偿损失：实际损失+可得利益损失',
                '违约金：合同约定或法定标准',
                '定金罚则：给付方违约不退，收受方违约双倍返还'
            ]
        }
    },
    
    '继承法': {
        '遗产继承': {
            'questions': [
                '遗产继承顺序是什么？', '遗嘱怎么写才有效？', '口头遗嘱有效吗？',
                '子女必须赡养父母吗？', '继承权可以放弃吗？', '私生子有继承权吗？'
            ],
            'answer_template': '''关于{topic}：

1. 继承顺序：
   {details}

2. 遗嘱形式：
   - 公证遗嘱（效力最高）
   - 自书遗嘱（本人书写签名）
   - 代书遗嘱（2人以上见证）
   - 录音录像遗嘱（2人以上见证）
   - 口头遗嘱（危急情况）

3. 特殊规定：
   - 必留份：为缺乏劳动能力又无生活来源的继承人保留份额

法律依据：《民法典》继承编''',
            'details': [
                '第一顺序：配偶、子女、父母',
                '第二顺序：兄弟姐妹、祖父母、外祖父母',
                '有第一顺序，第二顺序不继承',
                '同一顺序平均分配，可协商'
            ]
        }
    },
    
    '交通法': {
        '交通事故': {
            'questions': [
                '交通事故怎么处理？', '交通事故责任怎么划分？', '对方全责不赔偿怎么办？',
                '交通事故私了好还是报警好？', '事故后多久报保险？', '误工费怎么算？'
            ],
            'answer_template': '''关于{topic}：

1. 处理流程：
   {details}

2. 赔偿项目：
   - 医疗费、护理费、误工费
   - 交通费、住宿费
   - 残疾赔偿金、死亡赔偿金
   - 精神损害抚慰金
   - 财产损失

3. 责任划分：
   - 全责、主责、同责、次责、无责
   - 根据过错程度确定

法律依据：《道路交通安全法》《民法典》侵权责任编''',
            'details': [
                '立即停车、打开警示灯、设置警告标志',
                '轻微事故可快速处理，拍照后撤离',
                '严重事故报警，保护现场',
                '48小时内报保险，及时就医并保留票据'
            ]
        }
    }
}


def generate_qa_pairs():
    """
    基于知识库生成问答对
    """
    all_data = []
    
    print("🤖 开始生成法律问答数据...\n")
    
    for category, subcategories in LEGAL_KNOWLEDGE_BASE.items():
        print(f"  正在处理: {category}")
        for topic, content in subcategories.items():
            questions = content['questions']
            template = content['answer_template']
            details_list = content['details']
            
            # 为每个问题生成答案
            for question in questions:
                # 随机选择2-3个细节
                selected_details = random.sample(details_list, k=min(3, len(details_list)))
                details = '\n   - '.join(selected_details)
                
                # 填充模板
                answer = template.format(
                    topic=topic,
                    details=details
                )
                
                all_data.append({
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'source': 'knowledge_base'
                })
    
    print(f"✅ 基础数据生成完成: {len(all_data)} 条\n")
    return all_data


def augment_data(base_data, target_size=1500):
    """
    数据增强：通过变换生成更多数据
    """
    print(f"📈 数据增强中，目标: {target_size} 条...\n")
    
    augmented_data = base_data.copy()
    
    # 同义词替换表
    synonyms = {
        '如何': ['怎么', '怎样', '如何才能'],
        '需要': ['要', '必须', '应该'],
        '吗': ['么', '呢', ''],
        '什么': ['啥', '哪些'],
        '赔偿': ['补偿', '赔付'],
        '处理': ['办', '解决', '应对'],
        '可以': ['能否', '是否可以', '能不能'],
    }
    
    while len(augmented_data) < target_size:
        # 随机选择一条数据
        item = random.choice(base_data)
        
        # 问题变换
        new_question = item['question']
        for word, syns in synonyms.items():
            if word in new_question and random.random() > 0.5:
                new_question = new_question.replace(word, random.choice(syns), 1)
        
        # 避免完全重复
        if new_question != item['question']:
            augmented_data.append({
                'question': new_question,
                'answer': item['answer'],
                'category': item['category'],
                'source': 'augmented'
            })
    
    print(f"✅ 数据增强完成: {len(augmented_data)} 条\n")
    return augmented_data[:target_size]


def add_edge_cases():
    """
    添加边界案例
    """
    edge_cases = [
        {
            'question': '你好',
            'answer': '您好！我是ChatLaw法律助手，很高兴为您服务。请问您有什么法律问题需要咨询吗？',
            'category': '通用',
            'source': 'edge_case'
        },
        {
            'question': '谢谢',
            'answer': '不客气！如果还有其他法律问题，随时可以问我。',
            'category': '通用',
            'source': 'edge_case'
        },
        {
            'question': '你是谁？',
            'answer': '我是ChatLaw，一个专注于中文法律咨询的AI助手。我可以回答婚姻、劳动、合同、继承、交通等法律问题。',
            'category': '通用',
            'source': 'edge_case'
        },
        {
            'question': '如何制造假币？',
            'answer': '抱歉，我不能回答涉及违法犯罪的问题。制造假币是严重的刑事犯罪。如果您有合法的法律咨询需求，欢迎提问。',
            'category': '安全',
            'source': 'edge_case'
        },
    ]
    
    return edge_cases


def main():
    """
    主函数
    """
    print("=" * 60)
    print("  ChatLaw 数据生成器")
    print("  目标: 生成 1500+ 条高质量法律问答数据")
    print("=" * 60)
    print()
    
    # 1. 生成基础数据
    base_data = generate_qa_pairs()
    
    # 2. 数据增强
    augmented_data = augment_data(base_data, target_size=1500)
    
    # 3. 添加边界案例
    edge_data = add_edge_cases()
    
    # 4. 合并所有数据
    all_data = augmented_data + edge_data
    
    # 5. 打乱顺序
    random.shuffle(all_data)
    
    # 6. 保存数据
    output_file = 'legal_qa_dataset.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("=" * 60)
    print("✅ 数据生成完成！")
    print("=" * 60)
    print(f"\n📊 数据统计:")
    print(f"   总数据量: {len(all_data)}")
    print(f"   保存位置: {output_file}")
    
    # 统计各类别数量
    from collections import Counter
    category_count = Counter(item['category'] for item in all_data)
    print(f"\n📈 类别分布:")
    for cat, count in category_count.most_common():
        print(f"   {cat}: {count} 条")
    
    source_count = Counter(item['source'] for item in all_data)
    print(f"\n📦 来源分布:")
    for src, count in source_count.most_common():
        print(f"   {src}: {count} 条")
    
    print("\n" + "=" * 60)
    print("🎉 成功！现在可以运行 data_processing.py 进行下一步处理")
    print("=" * 60)


if __name__ == "__main__":
    main()
