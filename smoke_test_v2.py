import torch
from maclo_v2 import MACLOSiamNetV2, compute_task_losses, MACLOController


def main():
    modalities = ["ncct", "cbf", "cbv", "mtt", "tmax"]
    model = MACLOSiamNetV2(
        modality_names=modalities,
        num_classes=2,
        meta_dim=29,
    )
    model.train()

    batch, height, width = 2, 256, 256
    inputs = {m: torch.randn(batch, 1, height, width) for m in modalities}
    availability = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
    ], dtype=torch.bool)
    metadata = torch.randn(batch, 29)

    outputs = model(inputs, availability, metadata)
    targets = {
        "seg": torch.randint(0, 2, (batch, 1, height, width)).float(),
        "age": torch.tensor([3.5, 0.0]),
        "cls": torch.tensor([0, 1]),
    }
    masks = {
        "seg": torch.tensor([1, 1], dtype=torch.bool),
        "age": torch.tensor([1, 0], dtype=torch.bool),
        "cls": torch.tensor([1, 1], dtype=torch.bool),
    }

    losses = compute_task_losses(outputs, targets, masks)
    active = {task: loss for task, loss in losses.items() if masks[task].any()}

    controller = MACLOController()
    shared_params, unified_parts, stats = controller.compute_unified_gradients(
        active, model.shared_parameters()
    )

    total = torch.stack(list(active.values())).sum()
    total.backward()
    controller.apply_unified_gradients(shared_params, unified_parts)

    assert outputs["seg"].shape == (batch, 1, height, width)
    assert outputs["age"].shape == (batch,)
    assert outputs["cls"].shape == (batch, 2)
    assert all(
        torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.grad is not None
    )

    print("Smoke test passed.")
    print("Active tasks:", stats["tasks"])
    print("Cosine matrix:\n", stats["cosine_matrix"])
    print("MACLO weights:", stats["weights"])
    print("Negative-pair fraction:", stats["negative_pair_fraction"])


if __name__ == "__main__":
    main()
