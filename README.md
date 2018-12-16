# JBM-classifying-defected-parts
### Generate model for classifying the defected manufacturing parts using Transfer Learning on dataset from JBM Group.

- This example shows how to train an image classifier based on TensorFlow Hub module that computes image feature vectors. 

- Since the size of the dataset was small so training the model from scratch might not give good accuracy scores so the concept of **Transfer Learning** as used to train the model.

#### Inception Network Implementation
- By default, it uses the feature vectors computed by Inception V3 trained on ImageNet. The top layer receives as input a 2048-dimensional vector (assuming Inception V3) for each image. We train a softmax layer on top of this
representation. 

- Since the softmax layer contains 2 labels (defected & healthy), this corresponds to learning 2 + 2048*2 model parameters for the biases and weights.

#### Training Classifier
- The subfolder names under the ```./data``` directory are labels to classify, since we have two labels ```defected & healthy``` so the data is placed under these directories. 

  - Classifier can be trained by executing the below command.

    ```python ./scripts/retrain.py --image_dir ./data``` 
  
  - There is also option to tune the specific hyperparameters for training the model, below command can be executed to train the model by tuning hyperparameters.
  
    ```python ./scripts/retrain.py -h```
- The training can be visualized using the tensorboard by executing the below command.

  ```tensorboard --logdir /tmp/retrain_logs```
    
#### Testing the model

- Alredy trained model present in ```./tmp/inception_v3/``` directory can be used for testing on new images. Execute the below command.

  ```bash
  python ./scripts/label_image.py \
  --graph=./tmp/inception_v3/output_graph.pb \
  --labels=./tmp/inception_v3/output_labels.txt \
  --input_layer=Placeholder \
  --output_layer=final_result \
  --image={Path of the Image file}
  ```
  
  The above trained model gave accuracy of **85.7%** with following hyperparameters:
  
    - no.of steps: 4000
    - batch size: 100
    - learning rate: 0.01

- The above model was trainedusing **Inception Network** it can also be trained using other pretrained network architectures e.g. **NASNet-A** (https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1)

#### References

   https://www.tensorflow.org/hub/tutorials/image_retraining

   https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html
   
   https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3

   https://arxiv.org/abs/1310.1531

   https://towardsdatascience.com/tensorflow-on-mobile-tutorial-1-744703297267

