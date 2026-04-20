import numpy as np

class MaskHistoryManager:
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.history = []
        self.current_step = -1
        self.base_mask = None 

    def init_base(self, initial_mask):
        self.base_mask = initial_mask.copy()
        self.history.clear()
        self.current_step = -1

    def push(self, old_mask, new_mask, action_name):
        diff = old_mask != new_mask
        if not np.any(diff): return 
            
        changed_indices = np.where(diff)
        old_vals = old_mask[changed_indices]
        new_vals = new_mask[changed_indices]
        
        if self.current_step < len(self.history) - 1:
            self.history = self.history[:self.current_step + 1]
            
        patch = {
            'action': action_name,
            'indices': changed_indices,
            'old_vals': old_vals,
            'new_vals': new_vals
        }
        self.history.append(patch)
        
        if len(self.history) > self.max_steps:
            oldest_patch = self.history.pop(0)
            self.base_mask[oldest_patch['indices']] = oldest_patch['new_vals']
        else:
            self.current_step += 1

    def undo(self, current_mask_ref):
        if self.current_step >= 0:
            patch = self.history[self.current_step]
            current_mask_ref[patch['indices']] = patch['old_vals']
            self.current_step -= 1
            return True
        return False

    def redo(self, current_mask_ref):
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            patch = self.history[self.current_step]
            current_mask_ref[patch['indices']] = patch['new_vals']
            return True
        return False