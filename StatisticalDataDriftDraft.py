from scipy import stats
import numpy as np

class StatisticalDriftDetector:
    """Detect distribution changes in inputs"""
    
    def __init__(self, baseline_window=1000):
        self.baseline_stats = {}
        self.baseline_window = baseline_window
    
    def calculate_baseline(self, data: np.ndarray, feature_name: str):
        """Establish baseline statistics"""
        self.baseline_stats[feature_name] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'samples': data
        }
    
    def detect_drift(self, new_data: np.ndarray, feature_name: str) -> dict:
        """Detect if new data has drifted from baseline"""
        
        if feature_name not in self.baseline_stats:
            return {"drift_detected": False, "reason": "No baseline"}
        
        baseline = self.baseline_stats[feature_name]
        
        # 1. Kolmogorov-Smirnov Test (distribution similarity)
        ks_statistic, ks_pvalue = stats.ks_2samp(
            baseline['samples'], 
            new_data
        )
        
        # 2. Mean shift detection
        mean_shift = abs(np.mean(new_data) - baseline['mean']) / baseline['std']
        
        # 3. Variance change
        variance_ratio = np.std(new_data) / baseline['std']
        
        # Drift thresholds
        drift_detected = (
            ks_pvalue < 0.05 or  # Significant distribution change
            mean_shift > 3 or     # Mean shifted > 3 standard deviations
            variance_ratio > 2 or variance_ratio < 0.5  # Variance doubled or halved
        )
        
        return {
            "drift_detected": drift_detected,
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "mean_shift_sigma": mean_shift,
            "variance_ratio": variance_ratio,
            "severity": self._calculate_severity(ks_pvalue, mean_shift, variance_ratio)
        }
    
    def _calculate_severity(self, ks_pvalue, mean_shift, variance_ratio):
        """Calculate drift severity"""
        if ks_pvalue < 0.01 or mean_shift > 5:
            return "CRITICAL"
        elif ks_pvalue < 0.05 or mean_shift > 3:
            return "HIGH"
        elif mean_shift > 2:
            return "MEDIUM"
        else:
            return "LOW"
