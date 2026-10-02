import torch
from maclo_v2 import MACLOSiamNetV2, compute_task_losses, MACLOController


def main():
    modalities = ["ncct", "cbf", "cbv", "mtt", "tmax"]
    model = MACLOSiamNetV2(modality_names=modalities, num_classes=None)
    model.train()

    B, H, W = 2, 256, 256
    inputs = {m: torch.randn(B, 1, H, W) for m in modalities}
    availability = torch.tensor([
        [1, 0, 0, 0, 0],  # NCCT-only
        [0, 1, 1, 1, 1],  # CTP maps
    ], dtype=torch.bool)

    outputs = model(inputs, availability)
    targets = {
        "seg": torch.randint(0, 2, (B, 1, H, W)).float(),
        "age": torch.tensor([3.5, 0.0]),
        "cls": torch.zeros(B, dtype=torch.long),
    }
    masks = {
        "seg": torch.tensor([1, 1], dtype=torch.bool),
        "age": torch.tensor([1, 0], dtype=torch.bool),
        "cls": torch.tensor([0, 0], dtype=torch.bool),
    }

    losses = compute_task_losses(outputs, targets, masks)
    losses = {k: v for k, v in losses.items() if masks[k].any()}

    maclo = MACLOController()
    _ = maclo.task_gradients(losses, model.shared_parameters())
    sum(losses.values()).backward()
    stats = maclo.overwrite_shared_gradients(losses, model.shared_parameters())

    assert outputs["seg"].shape == (B, 1, H, W)
    assert outputs["age"].shape == (B,)
    assert torch.isfinite(outputs["seg"]).all()
    assert torch.isfinite(outputs["age"]).all()
    print("Smoke test passed.")
    print("Tasks:", stats["tasks"])
    print("Cosine matrix:", stats["cosine_matrix"])
    print("MACLO weights:", stats["weights"])


if __name__ == "__main__":
    main()
