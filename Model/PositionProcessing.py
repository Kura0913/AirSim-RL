import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionProcessing(nn.Module):
    def __init__(self):
        super(PositionProcessing, self).__init__()
        self.fc = nn.Linear(6, 32)  # 6 = 3 (drone) + 3 (target)

    def forward(self, drone_pos, target_pos):
        combined_pos = torch.cat((drone_pos, target_pos), dim=-1)
        diff_vector = target_pos - drone_pos
        initial_velocity = F.normalize(diff_vector, dim=-1)
        processed_pos = F.relu(self.fc(combined_pos))
        return torch.cat((initial_velocity, processed_pos), dim=-1)