
import torch
import torch.nn as nn


class MaternalRiskMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = MaternalRiskMLP()
    model.eval()

    # Test with four records and six features
    sample_input = torch.randn(4, 6)

    with torch.no_grad():
        output = model(sample_input)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("MATERNAL RISK MLP ARCHITECTURE")
    print("--------------------------------")
    print(model)
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Total parameters:", total_parameters)

    assert output.shape == (4, 3)

    print("--------------------------------")
    print("MLP ARCHITECTURE VERIFICATION PASSED")
