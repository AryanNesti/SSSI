# 
# Model Compression
Deep neural networks have achieved great success in many tasks like computer vision, nature launguage processing, speech processing. However, typical neural networks are both computationally expensive and energy-intensive, which can be difficult to be deployed on devices with low computation resources. Therefore, a natural thought is to perform model compression to reduce model size and accelerate model training/inference without losing performance significantly. Model compression techniques can be divided into two categories: pruning and quantization. The pruning methods explore the redundancy in the model weights and try to remove/prune the redundant and uncritical weights. Quantization refers to compress models by reducing the number of bits required to represent weights or activations. We further elaborate on the two methods, pruning and quantization, in the following chapters. Besides, the figure below visualizes the difference between these two methods.
<!-- Insert Image here -->

# Knowledge Distillation (KD)
Knowledge Distillation (KD) is proposed in Distilling the Knowledge in a Neural Network, the compressed model is trained to mimic a pre-trained, larger model. This training setting is also referred to as “teacher-student”, where the large model is the teacher and the small model is the student. KD is often used to fine-tune the pruned model.
<!-- Insert Image here -->

Neural networks typically produce class probabilities by using a “softmax” output layer that converts the logit, z$_i$, computed for each class into a probability, $q_i$, by comparing $z_i$ with the other logits.
$$q_i = \frac{exp(z_i/T)}{Σ_jexp(z_j/T)}$$
where T is a temperature that is normally set to 1. Using a higher value for T produces a softer probability distribution over classes.

In the simplest form of distillation, knowledge is transferred to the distilled model by training it on a transfer set and using a soft target distribution for each case in the transfer set that is produced by using the cumbersome model with a high temperature in its softmax. The same high temperature is used when training the distilled model, but after it has been trained it uses a temperature of 1.

The code for Knowledge Distillation used is:
```
for batch_idx, (data, target) in enumerate(train_loader):
   data, target = data.to(device), target.to(device)
   optimizer.zero_grad()
   y_s = model_s(data)
   y_t = model_t(data)
   loss_cri = F.cross_entropy(y_s, target)

   # kd loss
   p_s = F.log_softmax(y_s/kd_T, dim=1)
   p_t = F.softmax(y_t/kd_T, dim=1)
   loss_kd = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]

   # total loss
   loss = loss_cir + loss_kd
   loss.backward()
```
# LevelPruning
Unfortunatly we were unable to actually use Knowledge Distillation due it being a pytorch method while we are using a tensorflow class and we cannot switch to pytorch due to the capability of my device. My computer cannot import pytorch and therefore cannot use the pytorch method. This is where prunning comes in. Earlier we spoke about model compression and one of the best ways to compress a model is to prune the models. Here we implemented Levelpruning which is used for pytorch but also has a tensorflow implementation by NNI.

Therefore `{ 'sparsity': 0.8, 'op_types': ['default'] }`means that all layers with specified op_types will be compressed with the same 0.8 sparsity. When `pruner.compress()` called, the model is compressed with masks and after that you can normally fine tune this model and pruned weights won’t be updated which have been masked.
```
params = {
    'weight1': 0.1666,
    'weight2': 0.1666,
    'weight3': 0.1666,
    'weight4': 0.1666,
    'weight5': 0.1666,
    'weight6': 0.1666,
    'dropout_rate': 0.47,
    'learning_rate': 0.007,
}

weights = [params['weight1'],params['weight2'],params['weight3'],params['weight4'],params['weight5'],params['weight6']]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

from simple_multi_unet_model import multi_unet_model, jacard_coef  

metrics=['accuracy', jacard_coef]

import nni
import tensorflow as tf
def get_model():
    return multi_unet_model(dr=params['dropout_rate'], n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
adam = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: nni.report_intermediate_result(logs['accuracy'])
)
model.compile(optimizer=adam, loss=total_loss, metrics=metrics)
model.summary()

from nni.algorithms.compression.tensorflow.pruning import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()

history1 = model.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=10, 
                    callbacks=[callback],
                    validation_data=(X_test, y_test), 
                    shuffle=False)
loss, accuracy, jar = model.evaluate(X_test, y_test, verbose=1)
nni.report_final_result(accuracy)
```
The code above is the model being implemented and you can see that majority of it is similar to the previous model but here we took the best model from the Hyperparamter Optimization and pruned it with:
```
from nni.algorithms.compression.tensorflow.pruning import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```
This lead to getting us higher accuracy without having to run more epochs. The accuracy high was 80% while the average accuracy was 76%.

###### Training and Validation IoU vs Epoch
<!-- Image here -->

###### Training and Validation Loss vs Epoch
<!-- Image here -->

###### Training and Validation Percision vs Recall
<!-- Image here -->


<!-- Insert Image here -->
From the few images displayed above and the other images within the folder you can deduce that the model does a decent job predicting. From the `training and validation loss at each epoch` image, at the end, the validation loss is greater than the training loss. This may indicate that the model is underfitting. Underfitting occurs when the model is unable to accurately model the training data, and which in turn generates errors. However, unlike the previous milestone we have majority of the validation loss lower than the training loss. The `training and validation IoU at each epoch` for this expermient is about 4 points lower than the previous milestone where we have a mean of 40% IoU while the previous milestone had a mean of 48% IoU. Precision can be seen as a measure of quality, and recall as a measure of quantity. Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results. The `Training and Validation Percision vs Recall` shows that the model is pretty reasonable with such a smooth curve and it is much smoother than the previous milestone. It still becomes perfect between 0.3 and 0.4 recall, but like last time it start declining making it less reliable going downward. Please view milestone 3 branch to compare the graphs shown above. More of these images can be viewed from the `Images for milestone-4` folder.