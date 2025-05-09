"""
Blacksmithing - Semantic clustering and evaluation framework

Contains:
- FORGE: Feature-Oriented Robust Grouping Engine (clustering algorithm)
- Sledgehammer: Semantic clustering evaluation metrics

The package provides both clustering capabilities and evaluation metrics
for binary/categorical data using semantic pattern analysis,
following a metalworking-inspired architecture.
"""

import pandas as pd
import numpy as np
import math
import statistics
import warnings
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Optional
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, check_array

warnings.filterwarnings("ignore", category=RuntimeWarning)

class Sledgehammer:
    """Semantic clustering evaluation metrics using SLEDgeHammer method."""
    
    def __init__(self, 
                 W: List[float] = [0.3, 0.1, 0.5, 0.1], 
                 particular_threshold: Optional[float] = None,
                 aggregation: str = 'median'):
        """
        Initialize the SLEDgeHammer evaluator with custom weights and parameters.
        
        Args:
        W: Weights for Support/Length/Exclusivity/Difference indicators (default: [0.3, 0.1, 0.5, 0.1])
        particular_threshold: Threshold for feature particularization (None disables)
        aggregation: Score aggregation method ('harmonic', 'geometric', or 'median')
        
        Example:
        evaluator = Sledgehammer(W=[0.4, 0.1, 0.4, 0.1])
        """
        
        self.W = W
        self.particular_threshold = particular_threshold
        self.aggregation = aggregation
    
    @staticmethod
    def _particularize_descriptors(descriptors: pd.DataFrame, 
                                 particular_threshold: float = 1.0) -> pd.DataFrame:
        """Particularize descriptors based on support."""
        for feature in descriptors.columns:
            column = np.array(descriptors[feature])
            min_support = np.min(column)
            max_support = np.max(column)
            toremove = column < min_support + particular_threshold * (max_support - min_support)
            descriptors.loc[toremove, feature] = 0.0
        return descriptors
    
    @staticmethod
    def semantic_descriptors(X: Union[np.ndarray, pd.DataFrame], 
                           labels: np.ndarray, 
                           particular_threshold: Optional[float] = None,
                           report_form: bool = False) -> Union[pd.DataFrame, Dict]:
        """
        Compute feature support descriptors for each cluster.
        
        Args:
        X: Input data (n_samples × n_features)
        labels: Cluster assignments (n_samples,)
        particular_threshold: Particularization threshold (optional)
        report_form: If True, returns sorted descriptors per cluster as dict
        
        Returns:
        DataFrame of feature supports or dict of sorted descriptors
        
        Example:
        descriptors = evaluator.semantic_descriptors(X, labels, report_form=True)
        """
        
        n_clusters = max(labels) + 1
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
            
        support = X.groupby(labels).mean()
        
        if particular_threshold is not None:
            support = Sledgehammer._particularize_descriptors(support, particular_threshold)
            
        if report_form:
            report = {}
            for i in range(n_clusters):    
                report[i] = support.loc[i][support.loc[i] > 0].sort_values(ascending=False)
            return report
        return support
    
    def score_clusters(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      labels: np.ndarray,
                      aggregation: Optional[str] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Calculate S/L/E/D scores for each cluster.
        
        Args:
        X: Input data matrix
        labels: Cluster assignments
        aggregation: Override for instance aggregation method
        
        Returns:
        DataFrame with scores for each cluster (n_clusters × 4)
        
        Example:
        cluster_scores = evaluator.score_clusters(X, labels)
        """
        
        aggregation = aggregation or self.aggregation
        n_clusters = max(labels) + 1
        descriptors = self.semantic_descriptors(X, labels, self.particular_threshold).transpose()
        
        def mean_gt_zero(x): 
            return 0 if np.count_nonzero(x) == 0 else np.mean(x[x > 0])
        
        # Calculate all four components
        support_score = [mean_gt_zero(descriptors[cluster]) for cluster in range(n_clusters)]
        
        descriptor_set_size = np.array([np.count_nonzero(descriptors[cluster]) 
                                      for cluster in range(n_clusters)])
        avg_set_size = np.mean(descriptor_set_size[descriptor_set_size > 0])
        length_score = [0 if sz == 0 else 1.0/(1.0 + abs(sz - avg_set_size)) 
                       for sz in descriptor_set_size]
        
        descriptor_sets = np.array([frozenset(descriptors.index[descriptors[cluster] > 0]) 
                                  for cluster in range(n_clusters)])
        exclusive_sets = [descriptor_sets[cluster].difference(
            frozenset.union(*np.delete(descriptor_sets, cluster)))
            for cluster in range(n_clusters)]
        exclusive_score = [0 if len(descriptor_sets[cluster]) == 0 
                          else len(exclusive_sets[cluster])/len(descriptor_sets[cluster]) 
                          for cluster in range(n_clusters)]
        
        ordered_support = [np.sort(descriptors[cluster]) for cluster in range(n_clusters)]
        diff_score = [math.sqrt(np.max(np.diff(ordered_support[cluster]))) 
                     for cluster in range(n_clusters)]
        
        score = pd.DataFrame.from_dict({
            'S': [self.W[0] * s for s in support_score],
            'L': [self.W[1] * l for l in length_score],
            'E': [self.W[2] * e for e in exclusive_score],
            'D': [self.W[3] * d for d in diff_score]
        })
        
        if aggregation == 'harmonic':
            score = score.transpose().apply(statistics.harmonic_mean)
        elif aggregation == 'geometric':
            score = score.transpose().apply(statistics.geometric_mean)
        elif aggregation == 'median':
            score = score.transpose().apply(statistics.median)
        else:
            assert aggregation is None
            
        return score
    
    def score(self,
             X: Union[np.ndarray, pd.DataFrame],
             labels: np.ndarray) -> float:
        """
        Compute mean SLEDgeH score across all clusters.
        
        Args:
        X: Input data
        labels: Cluster assignments
        
        Returns:
        Average score (float)
        
        Example:
        print(f"Overall quality: {evaluator.score(X, labels):.3f}")
        """
        return np.mean(self.score_clusters(X, labels))
    
    def find_optimal_k(self,
                      data: Union[np.ndarray, pd.DataFrame],
                      clusterer: ClusterMixin,
                      k_max: int,
                      plot: bool = True) -> Dict[int, float]:
        """
        Evaluate clustering quality for different k values.
        
        Args:
            data: Input binary data
            clusterer: Clustering algorithm instance (must implement set_params and fit_predict)
            k_max: Maximum number of clusters to evaluate (must be ≥ 2)
            plot: Whether to plot results
            
        Returns:
            Dictionary of {k: mean_score}
            
        Raises:
            ValueError: If k_max is less than 2 or greater than n_samples
        """
        if k_max < 2:
            raise ValueError("k_max must be at least 2")
            
        n_samples = len(data)
        if k_max > n_samples:
            raise ValueError(f"k_max={k_max} cannot be greater than n_samples={n_samples}")
            
        results = {}
        
        for k in range(2, k_max + 1):
            try:
                clusterer.set_params(n_clusters=k)
                labels = clusterer.fit_predict(data)
                score = self.score(data, labels)
                results[k] = score
            except Exception as e:
                warnings.warn(f"Failed to cluster with k={k}: {str(e)}")
                results[k] = -1  # Indicate failure
                continue
        
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(results.keys(), results.values(), 'o-', color='royalblue')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Mean SLEDgeH Score')
            plt.title('Clustering Quality by Number of Clusters')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(2, k_max + 1))
            plt.show()
        
        return results


class FORGE(BaseEstimator, ClusterMixin):
    """FORGE (Feature-Oriented Robust Grouping Engine)
    
    A clustering algorithm for binary data using semantic dissimilarity metrics.
    """
    
    def __init__(self, n_clusters=2, evaluator=None, random_state=None):
        """
        Initialize FORGE clusterer.
        
        Args:
        n_clusters: Target number of clusters
        evaluator: Sledgehammer instance for dissimilarity
        random_state: Seed for reproducible exemplar selection
        
        Example:
        clusterer = FORGE(n_clusters=3, random_state=42)
        """
        
        self.n_clusters = n_clusters
        self.evaluator = evaluator or Sledgehammer()
        self.random_state = random_state
        
    def _compute_dissimilarity(self, target, reference):
        """
        Compute SLEDgeH dissimilarity between two samples.
        
        Args:
        target: Sample to evaluate
        reference: Exemplar sample
        
        Returns:
        Dissimilarity score (float)
        """
        target = np.atleast_2d(target)
        reference = np.atleast_2d(reference)
        X = np.vstack([reference, target])
        labels = np.zeros(len(X), dtype=int)
        labels[-1] = 1
        return self.evaluator.score(X, labels)
        
    def fit(self, X, y=None):
        """
        Compute clustering and store exemplars.
        
        Args:
        X: Input binary data
        y: Ignored (for scikit-learn compatibility)
        
        Returns:
        self (fitted clusterer)
        
        Raises:
        ValueError: If n_samples < n_clusters
        
        Example:
        clusterer.fit(X)
        """
        X = check_array(X, dtype=[np.int8, np.int16, np.int32, np.int64])
        self.X_ = X
        n_samples = X.shape[0]
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}")
            
        if self.n_clusters == 1:
            self.labels_ = np.zeros(n_samples, dtype=int)
            self.exemplars_ = [0]
            return self
            
        # Compute pairwise dissimilarity matrix
        self.dissimilarity_matrix_ = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                self.dissimilarity_matrix_[i,j] = self._compute_dissimilarity(X[i], X[j])
                self.dissimilarity_matrix_[j,i] = self.dissimilarity_matrix_[i,j]
        
        # Select initial exemplars
        self.exemplars_ = []
        remaining_points = set(range(n_samples))
        
        # First exemplar: maximum average dissimilarity
        avg_dissim = self.dissimilarity_matrix_.mean(axis=1)
        first_exemplar = np.argmax(avg_dissim)
        self.exemplars_.append(first_exemplar)
        remaining_points.remove(first_exemplar)
        
        # Subsequent exemplars: maximize minimum distance to existing exemplars
        for _ in range(1, self.n_clusters):
            if not remaining_points:
                # Handle case when we run out of distinct points
                break
                
            # Find point with maximum minimum dissimilarity to existing exemplars
            best_point = None
            max_min_dissim = -1
            
            for p in remaining_points:
                min_dissim = min(self.dissimilarity_matrix_[p, e] for e in self.exemplars_)
                if min_dissim > max_min_dissim:
                    max_min_dissim = min_dissim
                    best_point = p
            
            if best_point is None:
                # If all remaining points are identical, pick randomly
                rng = np.random.RandomState(self.random_state)
                best_point = rng.choice(list(remaining_points))
                
            self.exemplars_.append(best_point)
            remaining_points.remove(best_point)
        
        # If we couldn't find enough distinct exemplars, reduce n_clusters
        actual_n_clusters = len(self.exemplars_)
        if actual_n_clusters < self.n_clusters:
            warnings.warn(f"Could only find {actual_n_clusters} distinct exemplars. "
                         f"Reducing n_clusters from {self.n_clusters} to {actual_n_clusters}.")
            self.n_clusters = actual_n_clusters
        
        # Assign clusters
        labels = np.array([np.argmin([self.dissimilarity_matrix_[i, e] for e in self.exemplars_]) 
                         for i in range(n_samples)])
        
        # Renumber clusters consecutively
        unique_labels = np.unique(labels)
        self.labels_ = np.searchsorted(unique_labels, labels)
        return self
        
    def predict(self, X):
        """
        Assign new data to clusters.
        
        Args:
        X: New samples to predict
        
        Returns:
        Cluster labels (n_samples,)
        
        Example:
        new_labels = clusterer.predict(new_data)
        """
        check_is_fitted(self)
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = check_array(X, dtype=[np.int8, np.int16, np.int32, np.int64])
        
        return np.array([
            np.argmin([self._compute_dissimilarity(x, self.X_[e]) for e in self.exemplars_])
            for x in X
        ])
        
    def fit_predict(self, X, y=None):
        """
        Convenience method for fit + predict.
        
        Args:
        X: Input data
        y: Ignored
        
        Returns:
        Cluster labels
        
        Example:
        labels = clusterer.fit_predict(X)
        """
        return self.fit(X).labels_