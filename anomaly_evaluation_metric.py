import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from scipy import stats
import warnings
from collections import defaultdict
import time

class RATS_Metric:
    """
    RATS (Ranking-based Anomaly Time Series) Metric
    
    一种新的时序异常检测评估指标，具有以下特点：
    1. 区间异常评估，快速计算
    2. 高可解释性
    3. 高区分度，能区分随机模型
    4. 阈值无关
    5. 完美模型可达上界
    6. 对异常分数鲁棒
    """
    
    def __init__(self, n_random_trials: int = 100, random_seed: int = 42):
        """
        初始化RATS指标
        
        Args:
            n_random_trials: 随机模型比较的试验次数
            random_seed: 随机种子
        """
        self.n_random_trials = n_random_trials
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def _get_anomaly_intervals(self, y_true: np.ndarray) -> List[Tuple[int, int]]:
        """
        从真实标签中提取异常区间
        
        Args:
            y_true: 真实标签数组 (0: 正常, 1: 异常)
            
        Returns:
            异常区间列表 [(start, end), ...]
        """
        intervals = []
        start = None
        
        for i, label in enumerate(y_true):
            if label == 1 and start is None:
                start = i
            elif label == 0 and start is not None:
                intervals.append((start, i - 1))
                start = None
        
        # 处理序列末尾的异常
        if start is not None:
            intervals.append((start, len(y_true) - 1))
            
        return intervals
    
    def _compute_interval_scores(self, y_scores: np.ndarray, intervals: List[Tuple[int, int]]) -> np.ndarray:
        """
        计算每个异常区间的代表性分数
        
        Args:
            y_scores: 异常分数数组
            intervals: 异常区间列表
            
        Returns:
            每个区间的代表性分数
        """
        interval_scores = []
        
        for start, end in intervals:
            # 使用区间内的最大分数作为代表性分数
            # 这样可以捕获到区间内最强的异常信号
            interval_score = np.max(y_scores[start:end+1])
            interval_scores.append(interval_score)
            
        return np.array(interval_scores)
    
    def _compute_normal_scores(self, y_scores: np.ndarray, y_true: np.ndarray, 
                             sample_size: Optional[int] = None) -> np.ndarray:
        """
        采样正常点的分数
        
        Args:
            y_scores: 异常分数数组
            y_true: 真实标签数组
            sample_size: 采样大小，如果为None则使用所有正常点
            
        Returns:
            正常点的分数样本
        """
        normal_indices = np.where(y_true == 0)[0]
        normal_scores = y_scores[normal_indices]
        
        if sample_size is not None and len(normal_scores) > sample_size:
            # 随机采样正常点
            sampled_indices = np.random.choice(len(normal_scores), 
                                             size=sample_size, replace=False)
            normal_scores = normal_scores[sampled_indices]
            
        return normal_scores
    
    def _ranking_comparison(self, anomaly_scores: np.ndarray, 
                          normal_scores: np.ndarray) -> float:
        """
        计算异常分数相对于正常分数的排序优势
        
        Args:
            anomaly_scores: 异常区间代表性分数
            normal_scores: 正常点分数样本
            
        Returns:
            排序比较分数 (0-1之间)
        """
        if len(anomaly_scores) == 0:
            return 0.0
            
        if len(normal_scores) == 0:
            return 1.0
        
        # 计算每个异常分数超过的正常分数比例
        total_comparisons = 0
        total_wins = 0
        
        for anomaly_score in anomaly_scores:
            wins = np.sum(anomaly_score > normal_scores)
            total_wins += wins
            total_comparisons += len(normal_scores)
        
        if total_comparisons == 0:
            return 0.0
            
        return total_wins / total_comparisons
    
    def _generate_random_scores(self, length: int, distribution: str = 'uniform') -> np.ndarray:
        """
        生成随机分数用于基准比较
        
        Args:
            length: 分数序列长度
            distribution: 分布类型 ('uniform', 'normal', 'exponential')
            
        Returns:
            随机分数数组
        """
        if distribution == 'uniform':
            return np.random.uniform(0, 1, length)
        elif distribution == 'normal':
            scores = np.random.normal(0, 1, length)
            # 标准化到[0,1]
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        elif distribution == 'exponential':
            scores = np.random.exponential(1, length)
            return scores / (scores.max() + 1e-8)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
    def compute_rats_score(self, y_true: np.ndarray, y_scores: np.ndarray, 
                          comparison_models: Optional[List[np.ndarray]] = None) -> Dict:
        """
        计算RATS分数
        
        Args:
            y_true: 真实标签数组
            y_scores: 模型异常分数数组
            comparison_models: 其他模型的异常分数列表，用于比较
            
        Returns:
            包含各种指标的字典
        """
        # 输入验证
        if len(y_true) != len(y_scores):
            raise ValueError("y_true and y_scores must have the same length")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1")
        
        # 提取异常区间
        intervals = self._get_anomaly_intervals(y_true)
        
        if len(intervals) == 0:
            warnings.warn("No anomaly intervals found in y_true")
            return {
                'rats_score': 0.0,
                'vs_random_score': 0.5,
                'vs_models_score': None,
                'n_intervals': 0,
                'avg_interval_length': 0,
                'computation_time': 0.0
            }
        
        start_time = time.time()
        
        # 计算异常区间代表性分数
        interval_scores = self._compute_interval_scores(y_scores, intervals)
        
        # 采样正常点分数（为了计算效率）
        normal_sample_size = min(10000, np.sum(y_true == 0))
        normal_scores = self._compute_normal_scores(y_true, y_scores, normal_sample_size)
        
        # 计算vs正常点的排序分数
        ranking_score = self._ranking_comparison(interval_scores, normal_scores)
        
        # 计算vs随机模型的分数
        random_scores = []
        for _ in range(self.n_random_trials):
            for dist in ['uniform', 'normal', 'exponential']:
                random_y_scores = self._generate_random_scores(len(y_scores), dist)
                random_interval_scores = self._compute_interval_scores(random_y_scores, intervals)
                random_normal_scores = self._compute_normal_scores(random_y_scores, y_true, normal_sample_size)
                random_ranking = self._ranking_comparison(random_interval_scores, random_normal_scores)
                random_scores.append(random_ranking)
        
        vs_random_score = np.mean(np.array(random_scores) < ranking_score)
        
        # 计算vs其他模型的分数
        vs_models_score = None
        if comparison_models:
            model_scores = []
            for model_scores_array in comparison_models:
                if len(model_scores_array) != len(y_scores):
                    warnings.warn("Comparison model scores length mismatch, skipping")
                    continue
                
                model_interval_scores = self._compute_interval_scores(model_scores_array, intervals)
                model_normal_scores = self._compute_normal_scores(model_scores_array, y_true, normal_sample_size)
                model_ranking = self._ranking_comparison(model_interval_scores, model_normal_scores)
                model_scores.append(model_ranking)
            
            if model_scores:
                vs_models_score = np.mean(np.array(model_scores) < ranking_score)
        
        # 计算最终的RATS分数（加权组合）
        rats_score = 0.6 * ranking_score + 0.4 * vs_random_score
        
        computation_time = time.time() - start_time
        
        return {
            'rats_score': rats_score,
            'ranking_score': ranking_score,
            'vs_random_score': vs_random_score,
            'vs_models_score': vs_models_score,
            'n_intervals': len(intervals),
            'avg_interval_length': np.mean([end - start + 1 for start, end in intervals]),
            'computation_time': computation_time
        }
    
    def evaluate_multiple_models(self, y_true: np.ndarray, 
                               models_scores: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        评估多个模型并返回比较结果
        
        Args:
            y_true: 真实标签数组
            models_scores: 模型名称到异常分数的映射
            
        Returns:
            包含所有模型评估结果的DataFrame
        """
        results = []
        model_names = list(models_scores.keys())
        
        for model_name, scores in models_scores.items():
            # 获取其他模型的分数用于比较
            other_models = [models_scores[other] for other in model_names if other != model_name]
            
            result = self.compute_rats_score(y_true, scores, other_models)
            result['model_name'] = model_name
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('rats_score', ascending=False)
        return df

class AnomalyDataGenerator:
    """
    异常数据生成器，用于测试不同场景
    """
    
    @staticmethod
    def generate_synthetic_data(n_points: int = 1000, 
                              n_anomalies: int = 5,
                              anomaly_length_range: Tuple[int, int] = (10, 50),
                              noise_level: float = 0.1,
                              random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成时序异常数据
        
        Args:
            n_points: 时序长度
            n_anomalies: 异常区间数量
            anomaly_length_range: 异常区间长度范围
            noise_level: 噪声水平
            random_seed: 随机种子
            
        Returns:
            (时序数据, 异常标签)
        """
        np.random.seed(random_seed)
        
        # 生成基础时序（正弦波 + 趋势 + 噪声）
        t = np.linspace(0, 4*np.pi, n_points)
        ts = np.sin(t) + 0.1*t + np.random.normal(0, noise_level, n_points)
        
        # 初始化标签
        labels = np.zeros(n_points)
        
        # 随机选择异常区间位置
        available_positions = list(range(n_points))
        
        for _ in range(n_anomalies):
            if len(available_positions) < anomaly_length_range[0]:
                break
                
            # 随机选择异常长度
            anomaly_length = np.random.randint(anomaly_length_range[0], 
                                             min(anomaly_length_range[1], len(available_positions)) + 1)
            
            # 随机选择起始位置
            max_start = len(available_positions) - anomaly_length + 1
            if max_start <= 0:
                break
                
            start_idx = np.random.randint(0, max_start)
            start_pos = available_positions[start_idx]
            
            # 确保区间不重叠
            end_pos = min(start_pos + anomaly_length - 1, n_points - 1)
            
            # 添加异常
            labels[start_pos:end_pos+1] = 1
            
            # 在时序中引入异常模式
            anomaly_amplitude = np.random.uniform(2, 5)
            anomaly_pattern = np.random.choice(['spike', 'step', 'drift'])
            
            if anomaly_pattern == 'spike':
                ts[start_pos:end_pos+1] += anomaly_amplitude * np.random.choice([-1, 1])
            elif anomaly_pattern == 'step':
                ts[start_pos:end_pos+1] += anomaly_amplitude * np.random.choice([-1, 1])
            elif anomaly_pattern == 'drift':
                drift = np.linspace(0, anomaly_amplitude * np.random.choice([-1, 1]), end_pos - start_pos + 1)
                ts[start_pos:end_pos+1] += drift
            
            # 从可用位置中移除使用的位置
            available_positions = [pos for pos in available_positions 
                                 if pos < start_pos or pos > end_pos]
        
        return ts, labels

def create_test_models() -> Dict[str, callable]:
    """
    创建不同类型的测试模型
    """
    def perfect_model(ts, labels):
        """完美模型：在异常处给出高分，正常处给出低分"""
        scores = np.random.uniform(0, 0.2, len(ts))  # 基础低分
        scores[labels == 1] = np.random.uniform(0.8, 1.0, np.sum(labels == 1))  # 异常处高分
        return scores
    
    def good_model(ts, labels):
        """好模型：大部分异常能检测到，少量误报"""
        scores = np.random.uniform(0, 0.3, len(ts))
        anomaly_indices = np.where(labels == 1)[0]
        
        # 80%的异常被检测到
        detected = np.random.choice(anomaly_indices, size=int(0.8 * len(anomaly_indices)), replace=False)
        scores[detected] = np.random.uniform(0.7, 1.0, len(detected))
        
        # 5%的正常点被误报
        normal_indices = np.where(labels == 0)[0]
        false_positives = np.random.choice(normal_indices, size=int(0.05 * len(normal_indices)), replace=False)
        scores[false_positives] = np.random.uniform(0.6, 0.8, len(false_positives))
        
        return scores
    
    def mediocre_model(ts, labels):
        """中等模型：检测能力一般"""
        scores = np.random.uniform(0, 0.5, len(ts))
        anomaly_indices = np.where(labels == 1)[0]
        
        # 50%的异常被检测到
        detected = np.random.choice(anomaly_indices, size=int(0.5 * len(anomaly_indices)), replace=False)
        scores[detected] = np.random.uniform(0.5, 0.8, len(detected))
        
        return scores
    
    def random_model(ts, labels):
        """随机模型"""
        return np.random.uniform(0, 1, len(ts))
    
    def noise_robust_test(ts, labels, base_model_func, noise_std=0.1):
        """测试模型对噪声的鲁棒性"""
        base_scores = base_model_func(ts, labels)
        noise = np.random.normal(0, noise_std, len(base_scores))
        noisy_scores = base_scores + noise
        return np.clip(noisy_scores, 0, 1)
    
    return {
        'perfect_model': perfect_model,
        'good_model': good_model,
        'mediocre_model': mediocre_model,
        'random_model': random_model,
        'noise_robust_test': noise_robust_test
    }