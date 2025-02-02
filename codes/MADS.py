import numpy as np

class StepwiseMADS:
    def __init__(self, x0, bounds, delta0=1.0, delta_min=1e-3, alpha=2.0, gamma=0.5):
        self.x = np.array(x0)
        self.bounds = np.array(bounds)
        self.delta = delta0
        self.delta_min = delta_min
        self.alpha = alpha
        self.gamma = gamma
        self.mesh_size = delta0
        self.iteration = 0
        self.pending_eval = None
        self.best_value = None  # Store the best evaluation value
    
    def get_next_candidate(self):
        if self.pending_eval is not None:
            raise RuntimeError("Previous evaluation result is still pending.")
        
        directions = np.eye(len(self.x))
        candidates = [self.x + self.mesh_size * d for d in directions] + [self.x - self.mesh_size * d for d in directions]
        
        candidates = [np.clip(c, self.bounds[:, 0], self.bounds[:, 1]) for c in candidates]
        self.pending_eval = candidates.copy()
        return candidates
    
    def update_with_result(self, evaluations):
        if self.pending_eval is None:
            raise RuntimeError("No pending evaluations to update with.")

        best_idx = np.argmin(evaluations)
        best_candidate = self.pending_eval[best_idx]
        best_value = evaluations[best_idx]

        
        if self.best_value is None or best_value < self.best_value:
            self.x = best_candidate
            self.best_value = best_value
            self.mesh_size *= self.alpha
        else:
            self.mesh_size *= self.gamma
        
        self.mesh_size = max(self.mesh_size, self.delta_min)
        self.pending_eval = None
        self.iteration += 1
    
    def get_current_solution(self):
        return self.x


class MADS:
    def __init__(self, x0, initial_value, bounds, delta0=1.0, delta_min=1e-3, alpha=2.0, gamma=0.5):
        self.stp_MADS = StepwiseMADS(x0, bounds=bounds, delta0=delta0, delta_min=delta_min, alpha=alpha, gamma=gamma)
        self.candidates = self.stp_MADS.get_next_candidate()
        self.evaluations = []
        self.step_num = 0
        self.iter = 0
        self.stp_MADS.best_value = initial_value

    def step(self, evaluation):
        self.evaluations.append(evaluation)
        if self.candidates == []:
            # new MADS iteration
            self.iter += 1
            self.stp_MADS.update_with_result(self.evaluations)
            self.candidates = self.stp_MADS.get_next_candidate()
            
            self.evaluations = []
        else:
            # print("pending candidatas...")
            pass
        self.step_num += 1
        return self.candidates.pop(0)