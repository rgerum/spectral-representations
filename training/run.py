import numpy as np
from train import main


for seed in [1234, 2468, 3702]:
    model_type = "shallow_mlp"
    for dataset in ["mnist"]:
        for strength in [0, 0.1, 1]:
            for reg1value in np.arange(0.6, 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=50, seed=seed,
                     output=f"logs/{model_type}/{dataset}/seed-{seed}_reg1-{strength}_reg1value-{reg1value}")

    model_type = "deep_mlp"
    for dataset in ["mnist"]:
        for strength in [0, 0.1, 1]:
            for reg1value in np.arange(0.6, 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=50, seed=seed,
                     output=f"logs/{model_type}/{dataset}/seed-{seed}_reg1-{strength}_reg1value-{reg1value}")

    model_type = "cnn"
    for dataset in ["mnist", "cifar10"]:
        for strength in [0, 0.1, 1]:
            for reg1value in np.arange(0.6, 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=200, seed=seed,
                     output=f"logs/{model_type}/{dataset}/seed-{seed}_reg1-{strength}_reg1value-{reg1value}")
