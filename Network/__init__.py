from Network.ActionNetwork import ActionNetwork
from Network.ValueNetwork import ValueNetwork
from Network.ModifiedResnet import ModifiedResNet18

available_networks = {
    'ActionNetwork': ActionNetwork,
    'ValueNetwork': ValueNetwork,
    'ModifiedResNet18': ModifiedResNet18
}