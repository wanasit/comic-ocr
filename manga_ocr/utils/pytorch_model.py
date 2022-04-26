import torch
from torch.utils.data import DataLoader


def calculate_validation_loss(model, validation_dataset, batch_size=1) -> float:
    assert getattr(model, "compute_loss"), 'Unknown model'

    total_loss = 0
    total_count = len(validation_dataset)
    with torch.no_grad():
        valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for i_batch, batch in enumerate(valid_dataloader):
            loss = model.compute_loss(batch)
            total_loss += loss.item()

    return total_loss / total_count


def get_total_parameters_count(model: torch.nn.Module) -> int:
    total_parameters_count = 0

    for parameter in model.parameters():
        size = 1
        for d in parameter.shape:
            size *= d

        total_parameters_count += size

    return total_parameters_count
