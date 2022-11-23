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
    from resnet import main

for repeat in range(3):
    for strength in [0.0, 1, 10]:
        for reg1value in np.arange(0.6, 4.1, 0.2):
            reg = strength
            data_augmentation = False
            main(
                 reg_strength=strength, reg_target=reg1value, epochs=200, repeat=repeat, data_augmentation=False,
                 output=f"{output_folder}/repeat-{repeat}_reg-strength-{strength}_reg-target-{reg1value:.1f}_data_augmentation-{data_augmentation}")
