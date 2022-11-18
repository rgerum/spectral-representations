import re
import pandas as pd

text = """
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:418: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.

Learning rate:  0.001
Epoch 1/200
390/391 [============================>.] - ETA: 0s - loss: 1.6677 - sparse_categorical_accuracy: 0.4541

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 46s 89ms/step - loss: 1.6669 - sparse_categorical_accuracy: 0.4544 - val_loss: 1.4759 - val_sparse_categorical_accuracy: 0.5191 - lr: 0.0010
Learning rate:  0.001
Epoch 2/200
391/391 [==============================] - ETA: 0s - loss: 1.2655 - sparse_categorical_accuracy: 0.6002

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 1.2655 - sparse_categorical_accuracy: 0.6002 - val_loss: 1.5575 - val_sparse_categorical_accuracy: 0.5304 - lr: 0.0010
Learning rate:  0.001
Epoch 3/200
391/391 [==============================] - ETA: 0s - loss: 1.0921 - sparse_categorical_accuracy: 0.6639

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 85ms/step - loss: 1.0921 - sparse_categorical_accuracy: 0.6639 - val_loss: 1.8530 - val_sparse_categorical_accuracy: 0.4984 - lr: 0.0010
Learning rate:  0.001
Epoch 4/200
390/391 [============================>.] - ETA: 0s - loss: 0.9748 - sparse_categorical_accuracy: 0.7097

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 34s 87ms/step - loss: 0.9749 - sparse_categorical_accuracy: 0.7097 - val_loss: 1.1945 - val_sparse_categorical_accuracy: 0.6439 - lr: 0.0010
Learning rate:  0.001
Epoch 5/200
390/391 [============================>.] - ETA: 0s - loss: 0.8905 - sparse_categorical_accuracy: 0.7378

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 85ms/step - loss: 0.8904 - sparse_categorical_accuracy: 0.7379 - val_loss: 1.0406 - val_sparse_categorical_accuracy: 0.6900 - lr: 0.0010
Learning rate:  0.001
Epoch 6/200
390/391 [============================>.] - ETA: 0s - loss: 0.8296 - sparse_categorical_accuracy: 0.7632

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 34s 87ms/step - loss: 0.8293 - sparse_categorical_accuracy: 0.7631 - val_loss: 1.1129 - val_sparse_categorical_accuracy: 0.6664 - lr: 0.0010
Learning rate:  0.001
Epoch 7/200
390/391 [============================>.] - ETA: 0s - loss: 0.7790 - sparse_categorical_accuracy: 0.7805

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 34s 87ms/step - loss: 0.7788 - sparse_categorical_accuracy: 0.7806 - val_loss: 0.8618 - val_sparse_categorical_accuracy: 0.7560 - lr: 0.0010
Learning rate:  0.001
Epoch 8/200
391/391 [==============================] - ETA: 0s - loss: 0.7379 - sparse_categorical_accuracy: 0.7964

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 85ms/step - loss: 0.7379 - sparse_categorical_accuracy: 0.7964 - val_loss: 1.0823 - val_sparse_categorical_accuracy: 0.6924 - lr: 0.0010
Learning rate:  0.001
Epoch 9/200
390/391 [============================>.] - ETA: 0s - loss: 0.7086 - sparse_categorical_accuracy: 0.8049

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.7086 - sparse_categorical_accuracy: 0.8049 - val_loss: 0.9197 - val_sparse_categorical_accuracy: 0.7389 - lr: 0.0010
Learning rate:  0.001
Epoch 10/200
390/391 [============================>.] - ETA: 0s - loss: 0.6818 - sparse_categorical_accuracy: 0.8136

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.6819 - sparse_categorical_accuracy: 0.8135 - val_loss: 1.0532 - val_sparse_categorical_accuracy: 0.7079 - lr: 0.0010
Learning rate:  0.001
Epoch 11/200
390/391 [============================>.] - ETA: 0s - loss: 0.6559 - sparse_categorical_accuracy: 0.8222

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 34s 86ms/step - loss: 0.6558 - sparse_categorical_accuracy: 0.8223 - val_loss: 0.8388 - val_sparse_categorical_accuracy: 0.7707 - lr: 0.0010
Learning rate:  0.001
Epoch 12/200
391/391 [==============================] - ETA: 0s - loss: 0.6399 - sparse_categorical_accuracy: 0.8284

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 34s 86ms/step - loss: 0.6399 - sparse_categorical_accuracy: 0.8284 - val_loss: 0.9133 - val_sparse_categorical_accuracy: 0.7514 - lr: 0.0010
Learning rate:  0.001
Epoch 13/200
390/391 [============================>.] - ETA: 0s - loss: 0.6193 - sparse_categorical_accuracy: 0.8351

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.6191 - sparse_categorical_accuracy: 0.8352 - val_loss: 0.8276 - val_sparse_categorical_accuracy: 0.7761 - lr: 0.0010
Learning rate:  0.001
Epoch 14/200
390/391 [============================>.] - ETA: 0s - loss: 0.6041 - sparse_categorical_accuracy: 0.8407

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.6039 - sparse_categorical_accuracy: 0.8407 - val_loss: 0.9363 - val_sparse_categorical_accuracy: 0.7493 - lr: 0.0010
Learning rate:  0.001
Epoch 15/200
391/391 [==============================] - ETA: 0s - loss: 0.5918 - sparse_categorical_accuracy: 0.8460

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.5918 - sparse_categorical_accuracy: 0.8460 - val_loss: 0.8031 - val_sparse_categorical_accuracy: 0.7820 - lr: 0.0010
Learning rate:  0.001
Epoch 16/200
391/391 [==============================] - ETA: 0s - loss: 0.5772 - sparse_categorical_accuracy: 0.8511

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.5772 - sparse_categorical_accuracy: 0.8511 - val_loss: 1.0859 - val_sparse_categorical_accuracy: 0.7106 - lr: 0.0010
Learning rate:  0.001
Epoch 17/200
390/391 [============================>.] - ETA: 0s - loss: 0.5612 - sparse_categorical_accuracy: 0.8566

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 34s 85ms/step - loss: 0.5610 - sparse_categorical_accuracy: 0.8567 - val_loss: 0.9916 - val_sparse_categorical_accuracy: 0.7455 - lr: 0.0010
Learning rate:  0.001
Epoch 18/200
390/391 [============================>.] - ETA: 0s - loss: 0.5537 - sparse_categorical_accuracy: 0.8597

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.5539 - sparse_categorical_accuracy: 0.8597 - val_loss: 0.8418 - val_sparse_categorical_accuracy: 0.7761 - lr: 0.0010
Learning rate:  0.001
Epoch 19/200
391/391 [==============================] - ETA: 0s - loss: 0.5421 - sparse_categorical_accuracy: 0.8627

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.5421 - sparse_categorical_accuracy: 0.8627 - val_loss: 0.7373 - val_sparse_categorical_accuracy: 0.8147 - lr: 0.0010
Learning rate:  0.001
Epoch 20/200
390/391 [============================>.] - ETA: 0s - loss: 0.5326 - sparse_categorical_accuracy: 0.8675

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.5326 - sparse_categorical_accuracy: 0.8675 - val_loss: 0.7733 - val_sparse_categorical_accuracy: 0.7903 - lr: 0.0010
Learning rate:  0.001
Epoch 21/200
391/391 [==============================] - ETA: 0s - loss: 0.5259 - sparse_categorical_accuracy: 0.8674

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.5259 - sparse_categorical_accuracy: 0.8674 - val_loss: 0.7778 - val_sparse_categorical_accuracy: 0.7954 - lr: 0.0010
Learning rate:  0.001
Epoch 22/200
390/391 [============================>.] - ETA: 0s - loss: 0.5114 - sparse_categorical_accuracy: 0.8743

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.5115 - sparse_categorical_accuracy: 0.8743 - val_loss: 0.8056 - val_sparse_categorical_accuracy: 0.7986 - lr: 0.0010
Learning rate:  0.001
Epoch 23/200
390/391 [============================>.] - ETA: 0s - loss: 0.5039 - sparse_categorical_accuracy: 0.8775

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 85ms/step - loss: 0.5039 - sparse_categorical_accuracy: 0.8776 - val_loss: 0.7129 - val_sparse_categorical_accuracy: 0.8191 - lr: 0.0010
Learning rate:  0.001
Epoch 24/200
391/391 [==============================] - ETA: 0s - loss: 0.4973 - sparse_categorical_accuracy: 0.8781

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4973 - sparse_categorical_accuracy: 0.8781 - val_loss: 0.6531 - val_sparse_categorical_accuracy: 0.8325 - lr: 0.0010
Learning rate:  0.001
Epoch 25/200
390/391 [============================>.] - ETA: 0s - loss: 0.4930 - sparse_categorical_accuracy: 0.8829

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4933 - sparse_categorical_accuracy: 0.8828 - val_loss: 0.7338 - val_sparse_categorical_accuracy: 0.8115 - lr: 0.0010
Learning rate:  0.001
Epoch 26/200
390/391 [============================>.] - ETA: 0s - loss: 0.4853 - sparse_categorical_accuracy: 0.8823

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.4852 - sparse_categorical_accuracy: 0.8823 - val_loss: 0.8710 - val_sparse_categorical_accuracy: 0.7852 - lr: 0.0010
Learning rate:  0.001
Epoch 27/200
391/391 [==============================] - ETA: 0s - loss: 0.4760 - sparse_categorical_accuracy: 0.8873

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4760 - sparse_categorical_accuracy: 0.8873 - val_loss: 0.8994 - val_sparse_categorical_accuracy: 0.7836 - lr: 0.0010
Learning rate:  0.001
Epoch 28/200
391/391 [==============================] - ETA: 0s - loss: 0.4760 - sparse_categorical_accuracy: 0.8868

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.4760 - sparse_categorical_accuracy: 0.8868 - val_loss: 0.9404 - val_sparse_categorical_accuracy: 0.7673 - lr: 0.0010
Learning rate:  0.001
Epoch 29/200
390/391 [============================>.] - ETA: 0s - loss: 0.4614 - sparse_categorical_accuracy: 0.8929

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.4614 - sparse_categorical_accuracy: 0.8929 - val_loss: 0.9325 - val_sparse_categorical_accuracy: 0.7691 - lr: 3.1623e-04
Learning rate:  0.001
Epoch 30/200
390/391 [============================>.] - ETA: 0s - loss: 0.4672 - sparse_categorical_accuracy: 0.8908

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4673 - sparse_categorical_accuracy: 0.8907 - val_loss: 0.7129 - val_sparse_categorical_accuracy: 0.8156 - lr: 0.0010
Learning rate:  0.001
Epoch 31/200
390/391 [============================>.] - ETA: 0s - loss: 0.4576 - sparse_categorical_accuracy: 0.8943

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4574 - sparse_categorical_accuracy: 0.8943 - val_loss: 0.7814 - val_sparse_categorical_accuracy: 0.7998 - lr: 0.0010
Learning rate:  0.001
Epoch 32/200
391/391 [==============================] - ETA: 0s - loss: 0.4518 - sparse_categorical_accuracy: 0.8955

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4518 - sparse_categorical_accuracy: 0.8955 - val_loss: 0.8109 - val_sparse_categorical_accuracy: 0.8096 - lr: 0.0010
Learning rate:  0.001
Epoch 33/200
391/391 [==============================] - ETA: 0s - loss: 0.4493 - sparse_categorical_accuracy: 0.8957

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 32s 82ms/step - loss: 0.4493 - sparse_categorical_accuracy: 0.8957 - val_loss: 0.6561 - val_sparse_categorical_accuracy: 0.8334 - lr: 0.0010
Learning rate:  0.001
Epoch 34/200
391/391 [==============================] - ETA: 0s - loss: 0.4447 - sparse_categorical_accuracy: 0.8989

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 32s 82ms/step - loss: 0.4447 - sparse_categorical_accuracy: 0.8989 - val_loss: 0.7452 - val_sparse_categorical_accuracy: 0.8157 - lr: 3.1623e-04
Learning rate:  0.001
Epoch 35/200
390/391 [============================>.] - ETA: 0s - loss: 0.4366 - sparse_categorical_accuracy: 0.9017

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.4368 - sparse_categorical_accuracy: 0.9016 - val_loss: 0.7954 - val_sparse_categorical_accuracy: 0.8067 - lr: 0.0010
Learning rate:  0.001
Epoch 36/200
390/391 [============================>.] - ETA: 0s - loss: 0.4362 - sparse_categorical_accuracy: 0.9016

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4363 - sparse_categorical_accuracy: 0.9016 - val_loss: 0.8161 - val_sparse_categorical_accuracy: 0.8037 - lr: 0.0010
Learning rate:  0.001
Epoch 37/200
390/391 [============================>.] - ETA: 0s - loss: 0.4344 - sparse_categorical_accuracy: 0.9006

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 32s 82ms/step - loss: 0.4346 - sparse_categorical_accuracy: 0.9006 - val_loss: 0.6913 - val_sparse_categorical_accuracy: 0.8314 - lr: 0.0010
Learning rate:  0.001
Epoch 38/200
390/391 [============================>.] - ETA: 0s - loss: 0.4342 - sparse_categorical_accuracy: 0.9027

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.4343 - sparse_categorical_accuracy: 0.9027 - val_loss: 0.7287 - val_sparse_categorical_accuracy: 0.8288 - lr: 0.0010
Learning rate:  0.001
Epoch 39/200
391/391 [==============================] - ETA: 0s - loss: 0.4264 - sparse_categorical_accuracy: 0.9061

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4264 - sparse_categorical_accuracy: 0.9061 - val_loss: 0.7356 - val_sparse_categorical_accuracy: 0.8263 - lr: 3.1623e-04
Learning rate:  0.001
Epoch 40/200
390/391 [============================>.] - ETA: 0s - loss: 0.4290 - sparse_categorical_accuracy: 0.9037

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4293 - sparse_categorical_accuracy: 0.9035 - val_loss: 0.6434 - val_sparse_categorical_accuracy: 0.8411 - lr: 0.0010
Learning rate:  0.001
Epoch 41/200
390/391 [============================>.] - ETA: 0s - loss: 0.4194 - sparse_categorical_accuracy: 0.9074

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 85ms/step - loss: 0.4195 - sparse_categorical_accuracy: 0.9074 - val_loss: 0.6050 - val_sparse_categorical_accuracy: 0.8533 - lr: 0.0010
Learning rate:  0.001
Epoch 42/200
390/391 [============================>.] - ETA: 0s - loss: 0.4164 - sparse_categorical_accuracy: 0.9094

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 32s 82ms/step - loss: 0.4164 - sparse_categorical_accuracy: 0.9093 - val_loss: 0.6757 - val_sparse_categorical_accuracy: 0.8366 - lr: 0.0010
Learning rate:  0.001
Epoch 43/200
391/391 [==============================] - ETA: 0s - loss: 0.4149 - sparse_categorical_accuracy: 0.9092

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 32s 82ms/step - loss: 0.4149 - sparse_categorical_accuracy: 0.9092 - val_loss: 0.6982 - val_sparse_categorical_accuracy: 0.8250 - lr: 0.0010
Learning rate:  0.001
Epoch 44/200
391/391 [==============================] - ETA: 0s - loss: 0.4102 - sparse_categorical_accuracy: 0.9106

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 32s 82ms/step - loss: 0.4102 - sparse_categorical_accuracy: 0.9106 - val_loss: 0.7748 - val_sparse_categorical_accuracy: 0.8165 - lr: 0.0010
Learning rate:  0.001
Epoch 45/200
391/391 [==============================] - ETA: 0s - loss: 0.4157 - sparse_categorical_accuracy: 0.9086

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4157 - sparse_categorical_accuracy: 0.9086 - val_loss: 0.9044 - val_sparse_categorical_accuracy: 0.7880 - lr: 0.0010
Learning rate:  0.001
Epoch 46/200
390/391 [============================>.] - ETA: 0s - loss: 0.4062 - sparse_categorical_accuracy: 0.9133

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 83ms/step - loss: 0.4064 - sparse_categorical_accuracy: 0.9132 - val_loss: 0.7923 - val_sparse_categorical_accuracy: 0.8142 - lr: 3.1623e-04
Learning rate:  0.001
Epoch 47/200
391/391 [==============================] - ETA: 0s - loss: 0.4080 - sparse_categorical_accuracy: 0.9127

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.4080 - sparse_categorical_accuracy: 0.9127 - val_loss: 0.7110 - val_sparse_categorical_accuracy: 0.8296 - lr: 0.0010
Learning rate:  0.001
Epoch 48/200
391/391 [==============================] - ETA: 0s - loss: 0.4032 - sparse_categorical_accuracy: 0.9135

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 34s 85ms/step - loss: 0.4032 - sparse_categorical_accuracy: 0.9135 - val_loss: 0.7601 - val_sparse_categorical_accuracy: 0.8154 - lr: 0.0010
Learning rate:  0.001
Epoch 49/200
391/391 [==============================] - ETA: 0s - loss: 0.4017 - sparse_categorical_accuracy: 0.9144

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.4017 - sparse_categorical_accuracy: 0.9144 - val_loss: 0.6979 - val_sparse_categorical_accuracy: 0.8425 - lr: 0.0010
Learning rate:  0.001
Epoch 50/200
390/391 [============================>.] - ETA: 0s - loss: 0.3998 - sparse_categorical_accuracy: 0.9151

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 84ms/step - loss: 0.3997 - sparse_categorical_accuracy: 0.9151 - val_loss: 0.7477 - val_sparse_categorical_accuracy: 0.8147 - lr: 0.0010
Learning rate:  0.001
Epoch 51/200
391/391 [==============================] - ETA: 0s - loss: 0.3962 - sparse_categorical_accuracy: 0.9173

WARNING:tensorflow:Can save best model only with val_acc available, skipping.

391/391 [==============================] - 33s 85ms/step - loss: 0.3962 - sparse_categorical_accuracy: 0.9173 - val_loss: 0.7168 - val_sparse_categorical_accuracy: 0.8266 - lr: 3.1623e-04
Learning rate:  0.001
"""

data = []
for match in re.finditer(r"Epoch (\d*)/200.*\n391/391 .*sparse_categorical_accuracy: ([\.\d]*)", text):
    epoch, sparse_categorical_accuracy = match.groups()
    data.append(dict(epoch=epoch, sparse_categorical_accuracy=sparse_categorical_accuracy))
pd.DataFrame(data).to_csv("resnet_no_reg.csv")