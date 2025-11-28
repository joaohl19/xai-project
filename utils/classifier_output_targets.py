import torch

# Method that generalizes the generation of heatmaps to output targets
class ClassifierOutputTargets:
    def __init__(self, categories):
        self.categories = categories
    
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.sum(model_output[self.categories])
        return torch.sum(torch.cat([model_output[:, c] for c in self.categories], dim=0), dim=1)