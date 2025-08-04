#!/usr/bin/env python3
"""
RATS指标使用示例

这个脚本展示了如何在实际项目中使用RATS指标评估时序异常检测模型。
"""

import numpy as np
import pandas as pd
from anomaly_evaluation_metric import RATS_Metric, AnomalyDataGenerator

def simple_usage_example():
    """简单使用示例"""
    print("=== RATS指标简单使用示例 ===\n")
    
    # 1. 准备数据
    print("1. 生成示例数据...")
    generator = AnomalyDataGenerator()
    ts_data, true_labels = generator.generate_synthetic_data(
        n_points=500,
        n_anomalies=3,
        anomaly_length_range=(10, 25),
        random_seed=42
    )
    
    print(f"   数据长度: {len(ts_data)}")
    print(f"   异常点数: {np.sum(true_labels)}")
    
    # 2. 模拟模型异常分数
    print("\n2. 生成模型预测分数...")
    
    # 模拟一个较好的模型
    model_scores = np.random.uniform(0, 0.3, len(ts_data))  # 基础低分
    anomaly_indices = np.where(true_labels == 1)[0]
    # 在异常位置给出较高分数
    model_scores[anomaly_indices] = np.random.uniform(0.6, 1.0, len(anomaly_indices))
    
    print(f"   异常位置平均分数: {np.mean(model_scores[true_labels == 1]):.3f}")
    print(f"   正常位置平均分数: {np.mean(model_scores[true_labels == 0]):.3f}")
    
    # 3. 计算RATS分数
    print("\n3. 计算RATS分数...")
    rats_evaluator = RATS_Metric(n_random_trials=50, random_seed=42)
    
    result = rats_evaluator.compute_rats_score(true_labels, model_scores)
    
    print(f"   RATS总分: {result['rats_score']:.4f}")
    print(f"   排序分数: {result['ranking_score']:.4f}")
    print(f"   vs随机模型分数: {result['vs_random_score']:.4f}")
    print(f"   计算时间: {result['computation_time']:.4f}秒")
    
    return result

def compare_multiple_models():
    """比较多个模型的示例"""
    print("\n=== 多模型比较示例 ===\n")
    
    # 生成测试数据
    generator = AnomalyDataGenerator()
    ts_data, true_labels = generator.generate_synthetic_data(
        n_points=800,
        n_anomalies=4,
        random_seed=123
    )
    
    # 创建不同质量的模型分数
    models_scores = {}
    
    # 1. 优秀模型 - 90%的异常能被检测到
    print("1. 生成优秀模型分数...")
    excellent_scores = np.random.uniform(0, 0.2, len(ts_data))
    anomaly_indices = np.where(true_labels == 1)[0]
    detected_anomalies = np.random.choice(anomaly_indices, 
                                        size=int(0.9 * len(anomaly_indices)), 
                                        replace=False)
    excellent_scores[detected_anomalies] = np.random.uniform(0.8, 1.0, len(detected_anomalies))
    models_scores['优秀模型'] = excellent_scores
    
    # 2. 一般模型 - 60%的异常能被检测到
    print("2. 生成一般模型分数...")
    good_scores = np.random.uniform(0, 0.4, len(ts_data))
    detected_anomalies = np.random.choice(anomaly_indices, 
                                        size=int(0.6 * len(anomaly_indices)), 
                                        replace=False)
    good_scores[detected_anomalies] = np.random.uniform(0.6, 0.9, len(detected_anomalies))
    models_scores['一般模型'] = good_scores
    
    # 3. 较差模型 - 只能检测到30%的异常
    print("3. 生成较差模型分数...")
    poor_scores = np.random.uniform(0, 0.6, len(ts_data))
    detected_anomalies = np.random.choice(anomaly_indices, 
                                        size=int(0.3 * len(anomaly_indices)), 
                                        replace=False)
    poor_scores[detected_anomalies] = np.random.uniform(0.5, 0.8, len(detected_anomalies))
    models_scores['较差模型'] = poor_scores
    
    # 4. 随机模型
    print("4. 生成随机模型分数...")
    models_scores['随机模型'] = np.random.uniform(0, 1, len(ts_data))
    
    # 评估所有模型
    print("\n5. 评估所有模型...")
    rats_evaluator = RATS_Metric(n_random_trials=30, random_seed=42)
    results_df = rats_evaluator.evaluate_multiple_models(true_labels, models_scores)
    
    print("\n模型排名结果:")
    display_cols = ['model_name', 'rats_score', 'ranking_score', 'vs_random_score', 'vs_models_score']
    print(results_df[display_cols].round(4).to_string(index=False))
    
    return results_df

def real_world_usage_pattern():
    """真实世界使用模式示例"""
    print("\n=== 真实世界使用模式 ===\n")
    
    # 假设我们有真实的时序数据和标签
    print("1. 加载真实数据（这里用模拟数据代替）...")
    
    # 模拟真实数据的特点：更长的时序，更复杂的异常模式
    generator = AnomalyDataGenerator()
    real_ts, real_labels = generator.generate_synthetic_data(
        n_points=2000,
        n_anomalies=8,
        anomaly_length_range=(5, 60),  # 异常长度变化大
        noise_level=0.15,  # 更多噪声
        random_seed=456
    )
    
    print(f"   数据长度: {len(real_ts)}")
    print(f"   异常区间数: {len(np.where(np.diff(np.concatenate(([0], real_labels, [0]))) == 1)[0])}")
    print(f"   异常率: {np.sum(real_labels) / len(real_labels) * 100:.2f}%")
    
    # 2. 假设我们训练了一个模型并得到了异常分数
    print("\n2. 模型预测...")
    # 模拟一个实际模型的输出（带有一些噪声和不完美性）
    model_predictions = np.random.uniform(0.1, 0.4, len(real_ts))
    
    # 在异常位置给出更高的分数，但不是完美的
    anomaly_indices = np.where(real_labels == 1)[0]
    for idx in anomaly_indices:
        if np.random.random() < 0.75:  # 75%的异常被检测到
            model_predictions[idx] = np.random.uniform(0.6, 0.95)
    
    # 添加一些误报
    normal_indices = np.where(real_labels == 0)[0]
    false_positive_indices = np.random.choice(normal_indices, 
                                            size=int(0.08 * len(normal_indices)), 
                                            replace=False)
    model_predictions[false_positive_indices] = np.random.uniform(0.5, 0.8)
    
    # 3. 使用RATS评估
    print("\n3. RATS评估...")
    rats_evaluator = RATS_Metric(n_random_trials=100, random_seed=42)
    
    evaluation_result = rats_evaluator.compute_rats_score(real_labels, model_predictions)
    
    print("评估结果:")
    print(f"   RATS总分: {evaluation_result['rats_score']:.4f}")
    print(f"   - 排序分数: {evaluation_result['ranking_score']:.4f}")
    print(f"   - vs随机模型: {evaluation_result['vs_random_score']:.4f}")
    print(f"   异常区间数: {evaluation_result['n_intervals']}")
    print(f"   平均区间长度: {evaluation_result['avg_interval_length']:.1f}")
    print(f"   计算时间: {evaluation_result['computation_time']:.4f}秒")
    
    # 4. 解释结果
    print("\n4. 结果解释:")
    if evaluation_result['rats_score'] > 0.8:
        print("   ✓ 优秀的模型性能")
    elif evaluation_result['rats_score'] > 0.6:
        print("   ◯ 良好的模型性能")
    elif evaluation_result['rats_score'] > 0.4:
        print("   △ 一般的模型性能，需要改进")
    else:
        print("   ✗ 较差的模型性能，建议重新训练")
    
    if evaluation_result['vs_random_score'] > 0.9:
        print("   ✓ 模型明显优于随机猜测")
    elif evaluation_result['vs_random_score'] > 0.7:
        print("   ◯ 模型优于随机猜测")
    else:
        print("   ✗ 模型性能接近随机水平")
    
    return evaluation_result

def batch_evaluation_example():
    """批量评估示例"""
    print("\n=== 批量评估示例 ===\n")
    
    print("模拟评估多个数据集上的模型性能...")
    
    datasets = [
        {"name": "数据集A", "n_points": 1000, "n_anomalies": 5, "seed": 100},
        {"name": "数据集B", "n_points": 1500, "n_anomalies": 8, "seed": 200},
        {"name": "数据集C", "n_points": 800, "n_anomalies": 3, "seed": 300},
    ]
    
    generator = AnomalyDataGenerator()
    rats_evaluator = RATS_Metric(n_random_trials=50, random_seed=42)
    
    batch_results = []
    
    for dataset in datasets:
        print(f"\n处理 {dataset['name']}...")
        
        # 生成数据集
        ts, labels = generator.generate_synthetic_data(
            n_points=dataset['n_points'],
            n_anomalies=dataset['n_anomalies'],
            random_seed=dataset['seed']
        )
        
        # 模拟模型预测（这里使用一个中等质量的模型）
        scores = np.random.uniform(0, 0.3, len(ts))
        anomaly_indices = np.where(labels == 1)[0]
        detected = np.random.choice(anomaly_indices, 
                                  size=int(0.7 * len(anomaly_indices)), 
                                  replace=False)
        scores[detected] = np.random.uniform(0.6, 0.9, len(detected))
        
        # 评估
        result = rats_evaluator.compute_rats_score(labels, scores)
        
        batch_results.append({
            'dataset': dataset['name'],
            'rats_score': result['rats_score'],
            'ranking_score': result['ranking_score'],
            'vs_random_score': result['vs_random_score'],
            'n_intervals': result['n_intervals'],
            'computation_time': result['computation_time']
        })
        
        print(f"   RATS分数: {result['rats_score']:.4f}")
    
    # 汇总结果
    batch_df = pd.DataFrame(batch_results)
    print("\n批量评估汇总:")
    print(batch_df.round(4).to_string(index=False))
    
    print(f"\n平均RATS分数: {batch_df['rats_score'].mean():.4f}")
    print(f"最佳数据集: {batch_df.loc[batch_df['rats_score'].idxmax(), 'dataset']}")
    
    return batch_df

def main():
    """主函数"""
    print("RATS (Ranking-based Anomaly Time Series) 指标使用示例")
    print("=" * 60)
    
    # 1. 简单使用示例
    simple_result = simple_usage_example()
    
    # 2. 多模型比较
    comparison_result = compare_multiple_models()
    
    # 3. 真实世界使用模式
    real_world_result = real_world_usage_pattern()
    
    # 4. 批量评估
    batch_result = batch_evaluation_example()
    
    print("\n" + "=" * 60)
    print("使用示例总结")
    print("=" * 60)
    
    print("\nRATS指标的主要优势:")
    print("✓ 无需设置阈值参数")
    print("✓ 可以评估区间异常")
    print("✓ 对异常分数的噪声鲁棒")
    print("✓ 能够区分不同质量的模型")
    print("✓ 计算速度快，适合大规模应用")
    print("✓ 结果易于解释")
    
    print("\n建议的使用场景:")
    print("• 时序异常检测模型的性能评估")
    print("• 多个模型之间的性能比较")
    print("• 模型超参数调优中的目标函数")
    print("• 异常检测系统的持续监控")
    
    print(f"\n示例完成！")

if __name__ == "__main__":
    main()