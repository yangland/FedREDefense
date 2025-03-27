
import torch
from utils import *
import numpy as np
from scipy.stats import beta

class OnlineLambdaOptimizer:
    def __init__(self, lambda_init=0.1, tol=1e-5, device="cuda", min_step=0.1):
        self.device = device
        self.tol = tol
        self.min_step = min_step  # Predefined minimum step size
        self.lambda_low = torch.tensor(0.0, device=device)
        self.lambda_high = torch.tensor(1.0, device=device)
        self.lambda_best = torch.tensor(lambda_init, device=device)  # Start with initial lambda
        
        # Store last two (lambda, distance) pairs
        self.prev_lambda = None
        self.prev_rate = None

    def l2_distance(self, p1, p2):
        """Compute L2 distance."""
        return torch.norm(p1 - p2, p=2)
    
    def update(self, s_t, m=None, b=None):
        """Update lambda based on past performance, skipping first iteration."""
        s_t = flat_dict(s_t)
        
        if self.prev_lambda is None:  
            # First iteration, just store the values
            self.prev_lambda = self.lambda_best
            self.prev_rate = None  # No rate to compare in first iteration
            return self.lambda_best

        # For second and subsequent iterations, calculate distance
        if m is not None and b is not None:
            m = flat_dict(m)
            b = flat_dict(b)
            r_t = self.l2_distance(s_t, b)/self.l2_distance(m, b)  # Current distance

            # Compare current distance with previous distance
            if self.prev_rate is not None and self.prev_rate > r_t:
                # Current lambda was better → Adjust range towards it
                print("Current lambda was better ")
                self.lambda_low = self.prev_lambda
            else:
                # Previous lambda was better → Adjust range towards it
                print("Previous lambda was better")
                self.lambda_high = self.prev_lambda

            # Update lambda as midpoint of new range
            new_lambda_best = (self.lambda_low + self.lambda_high) / 2

            # Ensure the step size is larger than min_step
            step_size = torch.abs(new_lambda_best - self.prev_lambda)
            if step_size < self.min_step:
                # Increase step size to meet the minimum threshold
                step_size = self.min_step
                new_lambda_best = self.prev_lambda + step_size * torch.sign(new_lambda_best - self.prev_lambda)

            # Store for next iteration
            self.lambda_best = new_lambda_best
            self.prev_lambda = self.lambda_best
            self.prev_rate = r_t

        return self.lambda_best


# # Example Usage
# device = "cuda" if torch.cuda.is_available() else "cpu"
# optimizer = OnlineLambdaOptimizer(device=device)

# for t in range(10):  # Simulating 10 iterations with new inputs
#     n = 3  # Dimension
#     s_t = torch.rand(n, device=device)
#     A = torch.rand(n, device=device)
#     b = torch.rand(n, device=device)
    
#     lambda_opt = optimizer.update(s_t, A, b)
#     print(f"Iteration {t+1}: Optimal lambda = {lambda_opt.item():.5f}")


import numpy as np

class ThompsonSamplingDelayed:
    def __init__(self, alpha=1, beta=1, window_size=50, min_step=0.05):
        self.alpha = alpha          # Beta prior (α)
        self.beta = beta           # Beta prior (β)
        self.window_size = window_size  # Sliding window for non-stationarity
        self.min_step = min_step   # Minimum exploration step size
        self.lambda_history = []   # Track chosen lambdas
        self.accuracy_history = [] # Track observed accuracies
        self.pending_lambda = None # Stores lambda waiting for accuracy feedback

    def choose_lambda(self):
        # If no pending lambda, sample a new one
        if self.pending_lambda is None:
            lambda_sample = np.random.beta(self.alpha, self.beta)
            
            # Optional: Enforce minimum step from previous best lambda
            if len(self.lambda_history) > 0:
                prev_best = self.lambda_history[np.argmax(self.accuracy_history)]
                if abs(lambda_sample - prev_best) < self.min_step:
                    lambda_sample = prev_best + np.sign(lambda_sample - prev_best) * self.min_step
            
            self.pending_lambda = np.clip(lambda_sample, 0, 1)
        
        return self.pending_lambda

    def update(self, accuracy_t):
        if self.pending_lambda is None:
            raise ValueError("Call choose_lambda() before update()")
        
        # Store the pending lambda and its observed accuracy
        self.lambda_history.append(self.pending_lambda)
        self.accuracy_history.append(accuracy_t)
        
        # Apply sliding window (forget old data if needed)
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)
            self.lambda_history.pop(0)
        
        # Update Beta posterior (normalize accuracy to [0, 1])
        normalized_acc = (accuracy_t - np.min(self.accuracy_history)) / \
                       (np.max(self.accuracy_history) - np.min(self.accuracy_history) + 1e-6)
        self.alpha += normalized_acc
        self.beta += (1 - normalized_acc)
        
        # Reset pending lambda (ready for next iteration)
        self.pending_lambda = None
        
    
    
class MomentumThompsonSampling:
    def __init__(self, alpha=1, beta=1, window_size=50, momentum=0.8, min_step=0.05):
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.momentum = momentum  # Weight for past lambdas (0.8 = strong memory)
        self.min_step = min_step
        self.lambda_history = []
        self.accuracy_history = []
        self.pending_lambda = None
        self.ema_lambda = 0.5  # Tracks moving average of selected lambdas

    def choose_lambda(self):
        # Sample from Beta distribution
        raw_lambda = np.random.beta(self.alpha, self.beta)
        
        # Apply momentum: blend new sample with past trend
        smoothed_lambda = self.momentum * self.ema_lambda + (1 - self.momentum) * raw_lambda
        
        # Enforce minimum step from previous best (optional)
        if len(self.lambda_history) > 0:
            prev_best = self.lambda_history[np.argmax(self.accuracy_history)]
            if abs(smoothed_lambda - prev_best) < self.min_step:
                smoothed_lambda = prev_best + np.sign(smoothed_lambda - prev_best) * self.min_step
        
        self.pending_lambda = np.clip(smoothed_lambda, 0, 1)
        self.ema_lambda = smoothed_lambda  # Update EMA
        return self.pending_lambda

    def update(self, accuracy_t):
        self.lambda_history.append(self.pending_lambda)
        self.accuracy_history.append(accuracy_t)
        
        # Maintain window
        if len(self.accuracy_history) > self.window_size:
            self.lambda_history.pop(0)
            self.accuracy_history.pop(0)
        
        # Update Beta distribution (weight recent accuracies more)
        recent_acc = self.accuracy_history[-self.window_size:]
        normalized_acc = (accuracy_t - np.min(recent_acc)) / (np.max(recent_acc) - np.min(recent_acc) + 1e-6)
        self.alpha += 0.5 * normalized_acc  # Smaller updates for stability
        self.beta += 0.5 * (1 - normalized_acc)
        

class AttackThompsonSampling:
    def __init__(self, alpha=1, beta=1, window_size=50, momentum=0.8, min_step=0.05):
        self.alpha = alpha          # Beta prior (α)
        self.beta = beta            # Beta prior (β)
        self.window_size = window_size  
        self.momentum = momentum    # 0.9 = stronger memory
        self.min_step = min_step
        self.lambda_history = []    # Track chosen lambdas
        self.accuracy_history = []  # Track observed accuracies (lower = better)
        self.ema_lambda = 0.5       # Exponential moving average of λ
        self.pending_lambda = None   # λ waiting for accuracy feedback

    def choose_lambda(self):
        # Sample from Beta distribution
        raw_lambda = np.random.beta(self.alpha, self.beta)
        
        # Apply momentum toward historically bad λ's
        smoothed_lambda = self.momentum * self.ema_lambda + (1 - self.momentum) * raw_lambda
        
        # Enforce minimum exploration
        if len(self.lambda_history) > 0:
            prev_worst = self.lambda_history[np.argmin(self.accuracy_history)]  # λ with lowest accuracy
            if abs(smoothed_lambda - prev_worst) < self.min_step:
                smoothed_lambda = prev_worst + np.sign(smoothed_lambda - prev_worst) * self.min_step
        
        self.pending_lambda = np.clip(smoothed_lambda, 0, 1)
        self.ema_lambda = smoothed_lambda
        return self.pending_lambda

    def update(self, accuracy_t):
        self.lambda_history.append(self.pending_lambda)
        self.accuracy_history.append(accuracy_t)
        
        # Maintain sliding window
        if len(self.accuracy_history) > self.window_size:
            self.lambda_history.pop(0)
            self.accuracy_history.pop(0)
        
        # Update Beta to favor λ's that minimize accuracy
        windowed_acc = self.accuracy_history[-self.window_size:]
        worst_acc = np.min(windowed_acc)
        normalized_harm = (worst_acc - accuracy_t) / (worst_acc - np.max(windowed_acc) + 1e-6)  # How much worse this λ is
        
        self.alpha += 0.3 * normalized_harm  # Smaller, stable updates
        self.beta += 0.3 * (1 - normalized_harm)
        
        
# class DeltaAwareThompsonSampler:
#     def __init__(self, alpha=1, beta=1, window_size=30, momentum=0.85, min_step=0.03):
#         self.alpha = alpha
#         self.beta = beta
#         self.window_size = window_size
#         self.momentum = momentum
#         self.min_step = min_step
#         self.lambda_history = []
#         self.accuracy_history = []
#         self.delta_history = []
#         self.ema_lambda = 0.5
#         self.pending_lambda = None
#         self.initial_phase = True  # Tracks if we're in the warm-up period

#     def choose_lambda(self):
#         # Phase 1: Initial random sampling (first 2 iterations)
#         if self.initial_phase or len(self.delta_history) < 1:
#             base_lambda = np.random.beta(self.alpha, self.beta)
#             self.pending_lambda = np.clip(base_lambda, 0, 1)
#             return self.pending_lambda
        
#         # Phase 2: Normal delta-aware operation
#         raw_lambda = np.random.beta(self.alpha, self.beta)
#         smoothed_lambda = self.momentum * self.ema_lambda + (1 - self.momentum) * raw_lambda
        
#         # Safely get best_lambda (default to EMA if history is problematic)
#         best_lambda = self.ema_lambda  # Fallback value
#         if len(self.delta_history) > 0 and len(self.lambda_history) > 0:
#             try:
#                 best_idx = np.argmin(self.delta_history)
#                 best_lambda = self.lambda_history[best_idx]
#             except:
#                 best_lambda = self.ema_lambda
        
#         # Apply minimum step constraint
#         if abs(smoothed_lambda - best_lambda) < self.min_step:
#             direction = 1 if (smoothed_lambda - best_lambda) >= 0 else -1
#             smoothed_lambda = best_lambda + direction * self.min_step
        
#         self.pending_lambda = np.clip(smoothed_lambda, 0, 1)
#         self.ema_lambda = smoothed_lambda
#         return self.pending_lambda

#     def update(self, current_acc):
#         # Calculate delta if we have previous accuracy
#         if len(self.accuracy_history) > 0:
#             delta_acc = current_acc - self.accuracy_history[-1]
#             self.delta_history.append(delta_acc)
#             if len(self.delta_history) >= 2:  # Need at least 2 deltas for meaningful comparison
#                 self.initial_phase = False
        
#         self.lambda_history.append(self.pending_lambda)
#         self.accuracy_history.append(current_acc)
        
#         # Maintain sliding window
#         if len(self.accuracy_history) > self.window_size:
#             self.lambda_history.pop(0)
#             self.accuracy_history.pop(0)
#             if len(self.delta_history) > 0:
#                 self.delta_history.pop(0)
        
#         # Update Beta distribution
#         if len(self.delta_history) > 1:  # Need at least 2 deltas
#             worst_delta = np.max(self.delta_history)
#             best_delta = np.min(self.delta_history)
#             if (worst_delta - best_delta) > 1e-6:  # Check for meaningful range
#                 current_delta = self.delta_history[-1]
#                 delta_normalized = (worst_delta - current_delta) / (worst_delta - best_delta)
#                 reward = delta_normalized * (1.0 if current_delta <= 0 else 0.5)
#                 self.alpha = min(100, self.alpha + 0.4 * reward)  # Cap alpha/beta growth
#                 self.beta = min(100, self.beta + 0.4 * (1 - reward))


class DeltaAwareThompsonSampler:
    def __init__(self, alpha=1, beta=1, window_size=30, momentum=0.85, min_step=0.03, acc_threshold=0.3):
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.momentum = momentum
        self.min_step = min_step
        self.acc_threshold = acc_threshold  # Minimum accuracy to allow aggressive lambda changes
        
        self.lambda_history = []
        self.accuracy_history = []
        self.delta_history = []
        
        self.ema_lambda = 0.5
        self.pending_lambda = None
        self.first_update_done = False  # Ensures we have history before using delta-based selection

    def choose_lambda(self):
        # First call fallback
        if not self.first_update_done:
            print("First call detected - initializing lambda searcher")
            self.pending_lambda = np.clip(np.random.beta(self.alpha, self.beta), 0, 1)  # Ensure pending_lambda is set
            print(f"First chosen lambda: {self.pending_lambda}")  # Debugging output
            return self.pending_lambda

        # Sample raw lambda from Beta distribution
        raw_lambda = np.random.beta(self.alpha, self.beta)
        smoothed_lambda = self.momentum * self.ema_lambda + (1 - self.momentum) * raw_lambda

        if len(self.delta_history) > 0:
            best_idx = np.argmin(self.delta_history)
            best_lambda = self.lambda_history[best_idx]

            # Get current accuracy
            current_acc = self.accuracy_history[-1] if self.accuracy_history else 1.0

            # Dampening factor: Reduce lambda shifts when accuracy is low
            dampening = min(1.0, max(0.2, current_acc / self.acc_threshold))
            
            if abs(smoothed_lambda - best_lambda) < self.min_step * dampening:
                smoothed_lambda = best_lambda + np.sign(smoothed_lambda - best_lambda) * self.min_step * dampening

        # Store lambda values
        self.pending_lambda = np.clip(smoothed_lambda, 0, 1)
        print(f"Chosen lambda: {self.pending_lambda}")  # Debugging output
        self.ema_lambda = smoothed_lambda
        return self.pending_lambda

    def update(self, current_acc):
        if self.pending_lambda is None:
            raise ValueError("choose_lambda() must be called before update() to set pending_lambda.")

        # Calculate delta accuracy
        if len(self.accuracy_history) > 0:
            delta_acc = current_acc - self.accuracy_history[-1]
            self.delta_history.append(delta_acc)
            self.first_update_done = True  # Now we can use deltas for selection

        self.lambda_history.append(self.pending_lambda)
        self.accuracy_history.append(current_acc)

        # Maintain sliding window
        if len(self.accuracy_history) > self.window_size:
            self.lambda_history.pop(0)
            self.accuracy_history.pop(0)
            if len(self.delta_history) > 0:
                self.delta_history.pop(0)

        # Adaptive Beta update based on accuracy and deltas
        if len(self.delta_history) > 0:
            worst_delta = np.max(self.delta_history)
            delta_range = worst_delta - np.min(self.delta_history)

            if delta_range > 1e-6:
                delta_normalized = (worst_delta - self.delta_history[-1]) / delta_range

                # Scale reward based on current accuracy (low accuracy → smaller adjustments)
                accuracy_factor = min(1.0, max(0.1, current_acc / self.acc_threshold))
                reward = delta_normalized * accuracy_factor * (1.0 if self.delta_history[-1] <= 0 else 0.5)

                self.alpha += 0.4 * reward
                self.beta += 0.4 * (1 - reward)
