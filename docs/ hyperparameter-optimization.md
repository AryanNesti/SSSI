# Metis 
Metis is a tuner for hyperparameter optimization which I implemented into the Unet model for Semantic Segmentation of Satellite Imagery. Metis offers several benefits over other tuning algorithms. Most methods only predict the optimal configuration, but Metis will give you two outputs, a prediction for the optimal configuration along with a suggestion for the next trial. This comes in handy because you remove the process of guessing. 

Metis actually tells you if you need to resample a particular hyper-parameter, unlike other methods just assume training data has no noisy data. Metis even has a search strategy that balances exploration, exploitation, and possibly resampling.

<!-- Insert Image here -->

Through system benchmarking, Metis can collect data points that describe system inputs (for example, system workload and parameter values) and outputs (for example, performance metrics of interests) for training its model. When Metis does not change its prediction of the best performing configuration with additional data points, we say the model has converged. Maximizing the prediction accuracy and minimizing the model convergence time are two evaluation metrics for Metis.

Metis may sound so strong and powerful of a method however, there are limitations concerning the applicability of Metis to systems in general. 
- Support of different system parameter types. Some types of system parameters can be non-trivial to model with regression models
    1. Categorical parameters take on one of a fixed number of non-integer values such as boolean. Since categorical parameters are not continuous in nature, it can be difficult to model the relationship among possible values. However there is an implementation conceptually treats each categorical value as a new target system.
    2. Some systems have multi-step parameters, where one single configuration requires the system to go through a specific sequence of value changes for one or more parameters. Metis does not currently support multi-step parameters.
- Costs of changing system configurations. Applying configuration changes can incur costs for some systems.
    1. Server reboots might be necessary after a configuration change, thus service interruptions. To handle this case, system administrators can decide to push a configuration change only if the new configuration is predicted to offer a certain level of performance improvement (for example, 10% latency reduction). Administrators can also bound the cost of reconfiguration, for example, by performing reconfigurations gradually over time, or by bounding the parameter space exploration by the distance from the current running configuration.
    2. Mispredictions can result in system performance degradation. Fortunately, Gaussian Process offers two ways to gain insights regarding uncertainties. GP offers a confidence interval for each prediction, and a log-marginal likelihood score to quantify the model fitness with respect to the training dataset.


# Results 
For our experiment we ran: 
```
search_space = {
    'dropout_rate': {'_type': 'uniform', '_value': [0, 1]},
    'learning_rate': {'_type': 'uniform', '_value': [0.01, 0.1]},
}
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Metis'
experiment.config.tuner.class_args = {
    'optimize_mode': 'maximize'
}
experiment.config.max_trial_number = 5
experiment.config.trial_concurrency = 1
```
The search space are the Hyperparameters we will be comparing. Sadly due to the lack in power of my laptop I was forced to run only 5 trials which had a runtime of 40 minutes along with my change of epochs to 2 which can be seen below.

This is how we create and ran the model:
```
params = {
    'dropout_rate': 0.2,
    'learning_rate': 0.01,
}

optimized_params = nni.get_next_parameter()
print(optimized_params)
params.update(optimized_params)
print(params)

def get_model():
    return multi_unet_model(dr=params['dropout_rate'], n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
adam = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: nni.report_intermediate_result(logs['accuracy'])
)
model.compile(optimizer=adam, loss=total_loss, metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=2, 
                    callbacks=[callback],
                    validation_data=(X_test, y_test), 
                    shuffle=False)
loss, accuracy, jar = model.evaluate(X_test, y_test, verbose=1)
nni.report_final_result(accuracy)
```
This will run each hyperparameter that is recieved from `nni.get_next_parameter()` and we will compare these results locally which can be shown in the images below

<!-- Insert image here -->
This image was one of runs I did of the hyperparameters and you can see that most of the time you will get around 51% accuracy with the occasional low of 0.1% to 0.15%.

<!-- Insert image here -->
This next image also shows most of the time you will get around 51% accuracy, but there is a higher accuracy you can achieve. With a couple more runs, I saw a pattern of what lead to such a high accuracy

<!-- Insert image here -->

For 72% accuracy we had these hyperparameters:
```
weight1 0.14275048822974565
weight2 0.028572094712332064
weight3 0.030611461167412514
weight4 0.11564836785045343
weight5 0.11207893402160225
weight6 0.08448044736292522
dropout_rate 0.4620498546761752
learning_rate 0.006722484568397358
```
For the 73% accuracy we had these hyperparameters:
```
weight1 0.06531298209057146
weight2 0.07978854881356211
weight3 0.1482299935544838
weight4 0.06605000611751255
weight5 0.1081233855993154
weight6 0.15238977646369967
dropout_rate 0.47595711190911827
learning_rate 0.007620257665594932
```
We can conclude from this that the weights have little to no effect, however the droprate and learning rate are the important parameters. So i reran the program with the droprate around 0.47 and the learning rate of 0.007 to run the best model to geat the percision and recall graph along with the 10 best images. After running the program we recieved an accuracy of 72%. We also lowered the bach size to 8 which lead to better results than a batch size of 16.


###### Training and Validation IoU vs Epoch
<!-- Image here -->

###### Training and Validation Loss vs Epoch
<!-- Image here -->

###### Training and Validation Percision vs Recall
<!-- Image here -->


<!-- Insert Image here -->
From the few images displayed above and the other images within the folder you can deduce that the model does a decent job predicting. From the `training and validation loss at each epoch` image, at the end, the validation loss is greater than the training loss. This may indicate that the model is underfitting. Underfitting occurs when the model is unable to accurately model the training data, and which in turn generates errors. However, unlike the previous milestone we have majority of the validation loss lower than the training loss. The `training and validation IoU at each epoch` hor this expermient is about 10 points lower than the previous milestone where we have a mean of 44% IoU while the previous milestone had a mean of 48% IoU. Precision can be seen as a measure of quality, and recall as a measure of quantity. Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results. The `Training and Validation Percision vs Recall` shows that the model is pretty reasonable with such a smooth curve and it is much smoother than the previous milestone. Instead of at 0.6 recall the percision making it close ot perfect this time it becomes perfect between 0.3 and 0.4 recall, but like last time it start declining making it less reliable going downward. Please view milestone 2 branch to compare the graphs shown above. More of these images can be viewed from the `Images for milestone-3` folder.