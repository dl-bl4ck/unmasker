from torch.nn.functional import mse_loss


def completion_network_loss(input, target):
    return mse_loss(input,target)
