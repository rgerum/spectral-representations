import tensorflow as tf
import os
import numpy as np
#from keras.optimizers import Adam
#sparse_categorical...
from spectral_representations.resnet.resnet_definition import get_model, lr_schedule


from spectral_representations.logging import get_output_path
from spectral_representations.callbacks import SaveHistory
from spectral_representations.attacks import get_attack_metrics
from resnet_load_data import load_data

def main(output="logs_reg_strength-{reg_strength}_reg_target-{reg_target}_noaug",
         reg_strength=1,
         reg_target=4,
         data_augmentation=False,
         epochs=200,
         repeat=0,
         ):
    # set the seed depending on the repeat
    tf.random.set_seed((repeat + 1) * 1234)
    np.random.seed((repeat + 1) * 1234)

    output = output.replace("{reg_strength}", f"{reg_strength}")
    output = output.replace("{reg_target}", f"{reg_target}")
    # Training parameters
    batch_size = 128  # orig paper trained all networks with batch_size=128

    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)

    model = get_model(input_shape, depth, reg_strength=reg_strength, reg_target=reg_target)

    model.compile(loss='SparseCategoricalCrossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=lr_schedule(0)),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())
    model.summary()
    #print(model_type)

    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    cb = SaveHistory(get_output_path(main, locals()),
                     additional_logs_callback=[get_attack_metrics((x_test, y_test), [0.05])])#np.arange(0, 0.2, 0.01)

    history_logger = tf.keras.callbacks.CSVLogger(f"{output}.csv", separator=",")
    callbacks = [checkpoint, lr_reducer, lr_scheduler, history_logger]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save(f"{output}_model")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
