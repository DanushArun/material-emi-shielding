"""
User feedback collection and management system.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


class FeedbackCollector:
    """Collects and manages user feedback on predictions."""
    
    def __init__(self, feedback_path: str = "data/user_feedback.json"):
        """Initialize feedback collector."""
        self.feedback_path = feedback_path
        self.feedback_queue = []
        self.processed_feedback = []
        self.retraining_threshold = 100  # Trigger retraining after 100 new feedbacks
        self._load_existing_feedback()
    
    def _load_existing_feedback(self):
        """Load existing feedback data."""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, 'r') as f:
                    data = json.load(f)
                    self.feedback_queue = data.get('queue', [])
                    self.processed_feedback = data.get('processed', [])
            except Exception as e:
                print(f"Warning: Could not load feedback data: {e}")
    
    def _save_feedback(self):
        """Save feedback data to disk."""
        os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
        with open(self.feedback_path, 'w') as f:
            json.dump({
                'queue': self.feedback_queue,
                'processed': self.processed_feedback,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def submit_feedback(self,
                       prediction_id: str,
                       composition: Dict[str, float],
                       predicted_se: float,
                       actual_se: float,
                       frequency: float,
                       thickness: float,
                       conditions: Optional[Dict] = None,
                       user_notes: str = "") -> Dict:
        """Submit user feedback on a prediction."""
        feedback = {
            'id': f"feedback_{datetime.now().timestamp()}",
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'composition': composition,
            'predicted_se': predicted_se,
            'actual_se': actual_se,
            'frequency_mhz': frequency,
            'thickness_mm': thickness,
            'error': actual_se - predicted_se,
            'relative_error': abs(actual_se - predicted_se) / actual_se * 100,
            'conditions': conditions or {},
            'user_notes': user_notes,
            'status': 'pending_validation'
        }
        
        self.feedback_queue.append(feedback)
        self._save_feedback()
        
        # Calculate impact
        impact = self._estimate_feedback_impact(feedback)
        
        # Check if retraining needed
        needs_retraining = len(self.feedback_queue) >= self.retraining_threshold
        
        return {
            'feedback_id': feedback['id'],
            'impact': impact,
            'needs_retraining': needs_retraining,
            'queue_size': len(self.feedback_queue),
            'message': f"Thank you! Your measurement will improve predictions by ~{impact:.1%}"
        }
    
    def _estimate_feedback_impact(self, feedback: Dict) -> float:
        """Estimate the impact of this feedback on model accuracy."""
        # Higher impact for:
        # 1. Large errors (model was very wrong)
        # 2. Rare compositions (fills data gap)
        # 3. Extreme conditions (edge cases)
        
        error_impact = min(abs(feedback['error']) / 20, 1.0)  # Normalize to 0-1
        
        # Check if composition is rare
        composition_rarity = self._calculate_composition_rarity(feedback['composition'])
        
        # Check if conditions are extreme
        extreme_factor = 0.0
        if feedback['frequency_mhz'] > 10000 or feedback['frequency_mhz'] < 10:
            extreme_factor += 0.3
        if feedback['thickness_mm'] > 10 or feedback['thickness_mm'] < 0.1:
            extreme_factor += 0.3
        
        # Weighted impact
        impact = (0.4 * error_impact + 
                 0.4 * composition_rarity + 
                 0.2 * min(extreme_factor, 1.0))
        
        return impact
    
    def _calculate_composition_rarity(self, composition: Dict[str, float]) -> float:
        """Calculate how rare this composition is in our data."""
        if not self.processed_feedback:
            return 0.9  # Very valuable if we have little data
        
        # Count similar compositions
        similar_count = 0
        for feedback in self.processed_feedback:
            if self._compositions_similar(composition, feedback.get('composition', {})):
                similar_count += 1
        
        # More rare = higher value
        rarity = 1.0 - (similar_count / max(len(self.processed_feedback), 1))
        return max(0.1, rarity)  # At least 10% value
    
    def _compositions_similar(self, comp1: Dict[str, float], comp2: Dict[str, float], 
                            threshold: float = 0.1) -> bool:
        """Check if two compositions are similar."""
        for element in set(comp1.keys()) | set(comp2.keys()):
            if abs(comp1.get(element, 0) - comp2.get(element, 0)) > threshold * 100:
                return False
        return True
    
    def validate_feedback(self, feedback_id: str, is_valid: bool = True) -> bool:
        """Validate a feedback entry (for quality control)."""
        for i, feedback in enumerate(self.feedback_queue):
            if feedback['id'] == feedback_id:
                if is_valid:
                    feedback['status'] = 'validated'
                    self.processed_feedback.append(feedback)
                else:
                    feedback['status'] = 'rejected'
                
                self.feedback_queue.pop(i)
                self._save_feedback()
                return True
        
        return False
    
    def get_feedback_statistics(self) -> Dict:
        """Get statistics about collected feedback."""
        all_feedback = self.feedback_queue + self.processed_feedback
        
        if not all_feedback:
            return {
                'total_feedback': 0,
                'pending_validation': 0,
                'validated': 0,
                'average_error': None,
                'improvement_trend': None
            }
        
        errors = [f['error'] for f in all_feedback if 'error' in f]
        
        # Calculate improvement trend (are errors decreasing over time?)
        improvement_trend = None
        if len(errors) > 10:
            recent_errors = errors[-10:]
            older_errors = errors[-20:-10] if len(errors) > 20 else errors[:10]
            improvement = (np.mean(np.abs(older_errors)) - np.mean(np.abs(recent_errors))) / np.mean(np.abs(older_errors))
            improvement_trend = improvement * 100  # Percentage improvement
        
        return {
            'total_feedback': len(all_feedback),
            'pending_validation': len(self.feedback_queue),
            'validated': len(self.processed_feedback),
            'average_error': np.mean(np.abs(errors)) if errors else None,
            'rmse': np.sqrt(np.mean(np.square(errors))) if errors else None,
            'improvement_trend': improvement_trend,
            'contributors': len(set(f.get('user_id', 'anonymous') for f in all_feedback))
        }
    
    def get_high_impact_feedback_requests(self, n: int = 5) -> List[Dict]:
        """Identify high-impact materials to request feedback for."""
        # This will be enhanced when ML models are implemented
        # For now, return some strategic compositions
        
        requests = [
            {
                'composition': {'Fe': 70, 'C': 30},
                'reason': 'Common steel composition - high impact on industrial users',
                'expected_impact': 0.8
            },
            {
                'composition': {'Al': 95, 'Cu': 5},
                'reason': 'Aluminum alloy - limited data available',
                'expected_impact': 0.7
            },
            {
                'composition': {'Cu': 60, 'Ni': 40},
                'reason': 'Cupronickel - important for marine applications',
                'expected_impact': 0.75
            },
            {
                'composition': {'Fe': 50, 'Ni': 50},
                'reason': 'FeNi alloy - magnetic shielding applications',
                'expected_impact': 0.85
            },
            {
                'composition': {'C': 20, 'Polymer': 80},
                'reason': 'Carbon composite - emerging technology',
                'expected_impact': 0.9
            }
        ]
        
        return requests[:n]
    
    def prepare_retraining_data(self) -> Dict:
        """Prepare validated feedback for model retraining."""
        if not self.processed_feedback:
            return {
                'status': 'insufficient_data',
                'message': 'Need more validated feedback for retraining'
            }
        
        # Format data for training
        training_data = {
            'compositions': [],
            'frequencies': [],
            'thicknesses': [],
            'measured_se': [],
            'conditions': []
        }
        
        for feedback in self.processed_feedback:
            if feedback.get('status') == 'validated':
                training_data['compositions'].append(feedback['composition'])
                training_data['frequencies'].append(feedback['frequency_mhz'])
                training_data['thicknesses'].append(feedback['thickness_mm'])
                training_data['measured_se'].append(feedback['actual_se'])
                training_data['conditions'].append(feedback.get('conditions', {}))
        
        return {
            'status': 'ready',
            'n_samples': len(training_data['measured_se']),
            'data': training_data,
            'timestamp': datetime.now().isoformat()
        }