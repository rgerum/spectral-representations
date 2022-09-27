import numpy as np
import sys

#sys.argv.append("--plan")

if "--plan" in sys.argv:
    import pandas as pd
    data = []
    def main(**kwargs):
        print(kwargs)
        data.append(kwargs)
        pd.DataFrame(data).to_csv("jobs.csv")
else:
    from train import main

if 1:
 for repeat in [0]:
    model_type = "shallow_mlp"
    for dataset in ["mnist"]:
        for strength in [1.0]:#[0.0, 0.1, 1.0]:
            for reg1value in [2.0]:#np.arange(0.6, 5.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=50, repeat=repeat, min_x=0, max_x=1000,
                     output=f"logs3onlymseloss2/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}_B")
 exit()

for repeat in range(3):
    model_type = "shallow_mlp"
    for dataset in ["mnist"]:
        for strength in [0.0, 0.1, 1.0]:
            for reg1value in np.arange(0.6, 5.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=100, repeat=repeat, min_x=5, max_x=1000,
                     output=f"logs/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")

for repeat in range(3):
    model_type = "deep_mlp"
    for dataset in ["mnist"]:
        for strength in [0.0, 0.1, 1.0]:
            for reg1value in np.arange(0.6, 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=100, repeat=repeat, min_x=5, max_x=1000,
                     output=f"logs/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")

for repeat in range(3):
    model_type = "cnn"
    for dataset in ["mnist", "cifar10"]:
        for strength in [0.0, 0.1, 1.0]:
            for reg1value in np.arange(0.6, 4.1, 0.2):
                reg = strength
                main(dataset=dataset, model_type=model_type,
                     reg_strength=strength, reg_target=reg1value, epochs=100, repeat=repeat, min_x=5, max_x=1000,
                     output=f"logs/{model_type}/{dataset}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}")
