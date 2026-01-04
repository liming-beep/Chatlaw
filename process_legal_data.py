"""
法律问答数据处理脚本
专门用于处理 legal_qa_dataset.jsonl
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class LegalDataProcessor:
    def __init__(self, input_file='legal_qa_dataset.jsonl'):
        self.input_file = input_file
        self.data = []
        
    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("📥 加载数据")
        print("=" * 60)
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        print(f"✅ 成功加载 {len(self.data)} 条数据")
        print(f"   文件: {self.input_file}")
        
        # 显示数据样本
        if self.data:
            print(f"\n📋 数据样本:")
            sample = self.data[0]
            print(f"   问题: {sample['question'][:50]}...")
            print(f"   答案: {sample['answer'][:80]}...")
            print(f"   类别: {sample['category']}")
        
        return self.data
    
    def clean_data(self):
        """数据清洗"""
        print("\n" + "=" * 60)
        print("🧹 数据清洗")
        print("=" * 60)
        
        original_len = len(self.data)
        cleaned_data = []
        
        seen_questions = set()
        
        for item in self.data:
            q = item['question'].strip()
            a = item['answer'].strip()
            
            # 过滤规则
            if len(q) < 3:  # 问题太短
                continue
            if len(a) < 10:  # 答案太短
                continue
            if q in seen_questions:  # 去重
                continue
            
            seen_questions.add(q)
            cleaned_data.append(item)
        
        self.data = cleaned_data
        print(f"✅ 清洗完成: {original_len} -> {len(self.data)} 条")
        print(f"   去重: {original_len - len(self.data)} 条")
        
    def analyze_data(self):
        """数据分析"""
        print("\n" + "=" * 60)
        print("📊 数据分析")
        print("=" * 60)
        
        df = pd.DataFrame(self.data)
        
        print(f"\n总数据量: {len(df)}")
        
        print(f"\n类别分布:")
        category_counts = df['category'].value_counts()
        for cat, count in category_counts.items():
            percentage = count / len(df) * 100
            print(f"   {cat}: {count} 条 ({percentage:.1f}%)")
        
        print(f"\n来源分布:")
        source_counts = df['source'].value_counts()
        for src, count in source_counts.items():
            print(f"   {src}: {count} 条")
        
        # 长度统计
        df['q_len'] = df['question'].apply(len)
        df['a_len'] = df['answer'].apply(len)
        
        print(f"\n长度统计:")
        print(f"   问题长度: 平均 {df['q_len'].mean():.1f}, 中位数 {df['q_len'].median():.1f}")
        print(f"   答案长度: 平均 {df['a_len'].mean():.1f}, 中位数 {df['a_len'].median():.1f}")
        
        return df
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """划分数据集"""
        print("\n" + "=" * 60)
        print("✂️  数据集划分")
        print("=" * 60)
        
        df = pd.DataFrame(self.data)
        
        # 检查类别数量，确保每个类别至少有2个样本
        category_counts = df['category'].value_counts()
        small_categories = category_counts[category_counts < 2]
        
        if len(small_categories) > 0:
            print(f"\n⚠️  发现类别样本过少，将不使用分层采样:")
            for cat, count in small_categories.items():
                print(f"   {cat}: {count} 条")
            
            # 不使用分层采样
            train_val, test = train_test_split(
                df,
                test_size=test_ratio,
                random_state=42
            )
            
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_ratio_adjusted,
                random_state=42
            )
        else:
            # 使用分层采样
            print("\n✅ 使用分层采样，保持类别分布一致")
            
            train_val, test = train_test_split(
                df,
                test_size=test_ratio,
                stratify=df['category'],
                random_state=42
            )
            
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_ratio_adjusted,
                stratify=train_val['category'],
                random_state=42
            )
        
        # 保存数据集
        self.save_dataset(train, val, test)
        
        # 统计信息
        print("\n" + "=" * 60)
        print("📊 划分结果")
        print("=" * 60)
        print(f"训练集: {len(train):4d} 条 ({len(train)/len(df)*100:.1f}%)")
        print(f"验证集: {len(val):4d} 条 ({len(val)/len(df)*100:.1f}%)")
        print(f"测试集: {len(test):4d} 条 ({len(test)/len(df)*100:.1f}%)")
        print(f"总计:   {len(df):4d} 条")
        
        print("\n各类别在训练集中的分布:")
        train_dist = train['category'].value_counts()
        for cat, count in train_dist.items():
            print(f"   {cat}: {count} 条")
        
        return train, val, test
    
    def save_dataset(self, train, val, test):
        """保存数据集"""
        print("\n💾 保存数据集...")
        
        # 保存为jsonl格式
        train.to_json('train_data.jsonl', orient='records', lines=True, force_ascii=False)
        val.to_json('val_data.jsonl', orient='records', lines=True, force_ascii=False)
        test.to_json('test_data.jsonl', orient='records', lines=True, force_ascii=False)
        
        print("✅ 数据集已保存:")
        print("   - train_data.jsonl")
        print("   - val_data.jsonl")
        print("   - test_data.jsonl")
    
    def visualize_dataset(self):
        """数据可视化"""
        print("\n" + "=" * 60)
        print("📈 生成可视化图表")
        print("=" * 60)
        
        df = pd.DataFrame(self.data)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ChatLaw 数据集分析', fontsize=16, fontweight='bold')
        
        # 1. 类别分布（饼图）
        category_counts = df['category'].value_counts()
        colors = plt.cm.Set3(range(len(category_counts)))
        axes[0, 0].pie(category_counts.values, labels=category_counts.index, 
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('问题类别分布', fontsize=12, fontweight='bold')
        
        # 2. 类别分布（柱状图）
        category_counts.plot(kind='bar', ax=axes[0, 1], color='steelblue', edgecolor='black')
        axes[0, 1].set_title('各类别数量统计', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('类别')
        axes[0, 1].set_ylabel('数量')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 问题长度分布
        df['q_len'] = df['question'].apply(len)
        axes[1, 0].hist(df['q_len'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(df['q_len'].mean(), color='red', linestyle='--', 
                          label=f'平均: {df["q_len"].mean():.0f}')
        axes[1, 0].set_title('问题长度分布', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('字符数')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. 答案长度分布
        df['a_len'] = df['answer'].apply(len)
        axes[1, 1].hist(df['a_len'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(df['a_len'].mean(), color='red', linestyle='--',
                          label=f'平均: {df["a_len"].mean():.0f}')
        axes[1, 1].set_title('答案长度分布', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('字符数')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ 可视化图表已保存: data_analysis.png")
        
        # 创建数据集划分对比图
        self.plot_dataset_split()
    
    def plot_dataset_split(self):
        """绘制数据集划分对比图"""
        # 读取划分后的数据
        try:
            train = pd.read_json('train_data.jsonl', lines=True)
            val = pd.read_json('val_data.jsonl', lines=True)
            test = pd.read_json('test_data.jsonl', lines=True)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('数据集划分分析', fontsize=16, fontweight='bold')
            
            # 1. 数据集大小对比
            sizes = [len(train), len(val), len(test)]
            labels = ['训练集', '验证集', '测试集']
            colors = ['#FF9999', '#66B2FF', '#99FF99']
            
            bars = axes[0].bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
            axes[0].set_ylabel('数据量', fontsize=11)
            axes[0].set_title('数据集大小对比', fontsize=12, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar, size in zip(bars, sizes):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{size}\n({size/sum(sizes)*100:.1f}%)',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 2. 各数据集中的类别分布堆叠图
            categories = train['category'].unique()
            train_counts = [len(train[train['category']==cat]) for cat in categories]
            val_counts = [len(val[val['category']==cat]) for cat in categories]
            test_counts = [len(test[test['category']==cat]) for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.25
            
            axes[1].bar(x - width, train_counts, width, label='训练集', color='#FF9999', edgecolor='black')
            axes[1].bar(x, val_counts, width, label='验证集', color='#66B2FF', edgecolor='black')
            axes[1].bar(x + width, test_counts, width, label='测试集', color='#99FF99', edgecolor='black')
            
            axes[1].set_xlabel('类别', fontsize=11)
            axes[1].set_ylabel('数量', fontsize=11)
            axes[1].set_title('各类别在不同数据集中的分布', fontsize=12, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(categories, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('dataset_split_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ 数据集划分图表已保存: dataset_split_analysis.png")
            
        except FileNotFoundError:
            print("⚠️  请先运行 split_dataset() 生成数据集文件")


def main():
    """主函数"""
    print("\n" + "🎯" * 30)
    print("ChatLaw 数据处理流程")
    print("🎯" * 30)
    
    # 创建处理器
    processor = LegalDataProcessor(input_file='legal_qa_dataset.jsonl')
    
    # 1. 加载数据
    processor.load_data()
    
    # 2. 数据清洗
    processor.clean_data()
    
    # 3. 数据分析
    processor.analyze_data()
    
    # 4. 数据集划分
    train, val, test = processor.split_dataset()
    
    # 5. 数据可视化
    processor.visualize_dataset()
    
    print("\n" + "=" * 60)
    print("✨ 数据处理完成！")
    print("=" * 60)
    print("\n📁 输出文件:")
    print("   1. train_data.jsonl       - 训练数据")
    print("   2. val_data.jsonl         - 验证数据")
    print("   3. test_data.jsonl        - 测试数据")
    print("   4. data_analysis.png      - 数据分析图表")
    print("   5. dataset_split_analysis.png - 数据集划分图表")
    
    print("\n🚀 下一步:")
    print("   运行 model_training.py 开始模型训练")
    print("=" * 60)


if __name__ == "__main__":
    main()