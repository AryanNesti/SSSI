# Milestone 2
For this project we are using Unet which is a convolutional neural network meaning it is applied to visual images. In this case we will apply Unet to satelite imagery that were taken of several different locations ranging from cities to deserts and oceans. On the left is the image taken and the image on the right is the effect of Unet. 
<!-- Image here -->

![](https://...Dark.png)  |  ![](https://...Ocean.png)
![Figure_1](https://user-images.githubusercontent.com/98928740/200205279-83f298a4-5592-41a7-91c8-15774bfcbc52.png)
![Figure_2](https://user-images.githubusercontent.com/98928740/200205260-f1abc72a-ac6a-4091-a582-97a38e67fd38.png)
```
Building: #3C1098
Land (unpaved area): #8429F6
Road: #6EC1E4
Vegetation: #FEDD3A
Water: #E2A929
Unlabeled: #9B9B9B
```
These are the hexadecimal colors catogorized by what is located at a specific frame. The models will take a coordinate predict what is located there and will assign the color. However the model doesn't produce RGB colors and thats where we create labels which replace the RGB with integers. This is presented as:
```
def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg
```

The model used for this project is:
```
history1 = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)
```
you can find this model saved in the `models` folder incase you would like to work with the same model. We also tested another model however it was not saved but you can see the details here
```
history2=model_resnet_backbone.fit(X_train_prepr, 
          y_train,
          batch_size=16, 
          epochs=10,
          verbose=1,
          validation_data=(X_test_prepr, y_test))
```
###### Training and Validation IoU vs Epoch
<!-- Image here -->
![Training and validation IoU](https://user-images.githubusercontent.com/98928740/200205020-c3899c45-3779-4944-8c5b-fbb5ce44c223.png)
###### Training and Validation Loss vs Epoch
<!-- Image here -->
![Training and Validation loss](https://user-images.githubusercontent.com/98928740/200205031-f9d81a09-2a19-4009-9659-96467b574151.png)
###### Training and Validation Percision vs Recall
<!-- Image here -->
![Percision and Recall](https://user-images.githubusercontent.com/98928740/200205196-ef73273e-bc40-4df3-87ca-b2cb506f9920.png)



<!-- Images here -->
From the few images displayed above and the other images within the folder you can deduce that the model does a decent job predicting. From the `training and validation loss at each epoch` image, at times, the validation loss is greater than the training loss. This may indicate that the model is underfitting. Underfitting occurs when the model is unable to accurately model the training data, and which in turn generates errors. The `training and validation IoU at each epoch` shows the intersection over Union - "a metric used to evaluate Deep Learning algorithms by estimating how well a predicted mask or bounding box matches the ground truth data". This is not bad with the ammount of epochs used and with more eppochs possible get a better prediction at the cost of the run time however. 
