import numpy as np
import sys
import datetime

output_folder = f"logs_{datetime.datetime.today().strftime('%Y-%m-%d')}"

if "--plan" in sys.argv:
    import pandas as pd
    data = []
    def main(**kwargs):
        print(kwargs)
        data.append(kwargs)
        pd.DataFrame(data).to_csv("jobs.csv")
else:
    from train import main

for repeat in range(5):
    """
    model_type = "shallow_mlp"
    for dataset in ["mnist"]:
        for strength in [0.0, 1, 10]:
            for reg1value in np.arange(0., 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=100, repeat=repeat, min_x=0, max_x=10,
                     output=f"{output_folder}/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")

    model_type = "deep_mlp"
    for dataset in ["mnist"]:
        for strength in [0.0, 1, 10]:
            for reg1value in np.arange(0., 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=100, repeat=repeat, min_x=0, max_x=10,
                     output=f"{output_folder}/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")

    model_type = "cnn"
    for dataset in ["mnist", "cifar10"]:
        for strength in [0.0, 1, 10]:
            for reg1value in np.arange(0., 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=100, repeat=repeat, min_x=0, max_x=10,
                     output=f"{output_folder}/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")

    model_type = "deep-cnn"
    for dataset in ["mnist", "cifar10"]:
        for strength in [0.0, 1, 10]:
            for reg1value in np.arange(0., 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=100, repeat=repeat, min_x=0, max_x=10,
                     output=f"{output_folder}/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")
    """
    model_type = "cnn"
    for dataset in ["cifar10"]:
        for strength in [0.0, 1, 10]:
            for reg1value in np.arange(0., 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=1000, repeat=repeat, min_x=0, max_x=10,
                     output=f"{output_folder}/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")