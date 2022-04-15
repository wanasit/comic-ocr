from torch import nn


def get_total_parameters_count(model: nn.Module) -> int:
    total_parameters_count = 0

    for parameter in model.parameters():
        size = 1
        for d in parameter.shape:
            size *= d

        total_parameters_count += size

    return total_parameters_count
