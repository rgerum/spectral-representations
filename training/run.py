import numpy as np
from train import main


for repeat in range(2):
    model_type = "shallow_mlp"
    for dataset in ["mnist"]:
        for strength in [0.0, 0.1, 1.0]:
            for reg1value in np.arange(0.6, 5.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=50, repeat=repeat,
                     output=f"logs/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")

for repeat in range(2):
    model_type = "deep_mlp"
    for dataset in ["mnist"]:
        for strength in [0.0, 0.1, 1.0]:
            for reg1value in np.arange(0.6, 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=50, repeat=repeat,
                     output=f"logs/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")

for repeat in range(3):
    model_type = "cnn"
    for dataset in ["mnist", "cifar10"]:
        for strength in [0.0, 0.1, 1.0]:
            for reg1value in np.arange(0.6, 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=200, repeat=repeat,
                     output=f"logs/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")
