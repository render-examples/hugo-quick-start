---
title: "Production Machine Learning (14 May 2023)"
date: 2023-05-14T10:15:27+08:00
tags: ['blog']
---

# Production Machine Learning

For the past two weeks, I’ve been picking up Kaggle courses as well as Coursera courses. Here are some notes taken that I think might be useful:

Adapting to Data:
Different kind of data changes

- change in distribution
- change in depedencies and change in ingested data
- Code smell
- Model not updated to new data (Cold start problem)
    - dynamic train
    - understand model limit
- Reroll old model with model versioning
- Concept drift
    - Change in P(Y|X) is a shift in the underlying relationship between model input and output
- Data drift
    - Change in P(X) is a shift in the distribution of data
- Prediction Shift (Population)
    - Change P(X|Y) is a shift in model prediction
- Output shift (Co-variate Shift)

## Tuning Performance to reduce training time

| Constraint | Input/Output | CPU | Memory |
| --- | --- | --- | --- |
| Commonly Occurs | - Large inputs
- Input requires parsing
- Small models | - Expensive Computation
- Underpowered Hardware | - Large number of inputs
- complex models |
| Take Action | - Store efficiently
- Paralleize reads
Consider batch size | - Train on faster accel.
- Upgrade processor
- Run on TPU
- Simplify model | - Add more memory
- Use fewer layers
- Reduce batch size |
|  |  |  |  |

tensorflow.distribute.strategy

- mirrored
- multi-worker mirrored
- tpu
- parameter server

`tf.distribute`

1. Create a strategy object
    
    `strategy = tf.distribute.MultiWorkerMirroredStrategy()` 
    
2. Wrap creation of model parameters within strategy scope
    
    ```python
    with strategy.scope():
    	model = create_model()
    	model.compile(
    		loss = 'sparse_categorical_crossentropy'
    		optimizer = tf.keras.optimizers.Adam(0.0001),
    		metrics=['accuracy'])
    ```
    
3. Scale the batch size by the number of replicas in the cluster
    
    ```python
    per_replica_batch_size = 64
    global_batch_size = per_replica_batch_size \
    	* strategy.num_replicas_in_sync
    ```
    

Readings: Designing High-pe

Readings: Designing High-peformance ML Systems

In this module, you focus on either I/O performance or computational speed, depending on the

model. For more information, see the following readings and videos.

[● How to Evaluate the Performance of Your Machine Learning Model](https://www.kdnuggets.com/2020/09/performance-machine-learning-model.html)

[● Best practices for performance and cost optimization for machine learning](https://cloud.google.com/solutions/machine-learning/best-practices-for-ml-performance-cost)

[● How To Improve Machine Learning Model Performance: Five Ways](https://www.anolytics.ai/blog/how-to-improve-machine-learning-model-performance/)

[● Distributed TensorFlow model training on Cloud AI Platform (TF Dev Summit '20)](https://youtu.be/I29_VZ82KW4)

[● Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training)

[● Speeding Up Neural Network Training with Data Echoing](http://ai.googleblog.com/2020/05/speeding-up-neural-network-training.html)

[● Machine Learning Performance Improvement Cheat Sheet](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)

[● Building a High-Performance Data Pipeline with Tensorflow 2.x](https://www.notion.so/Production-Machine-Learning-5c7159c026fa4a31ae3723edcabbbd59?pvs=21)

[● Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training)

[● AutoML Tables](https://cloud.google.com/automl-tables)

[Kubeflow](https://www.kubeflow.org/)

[● Introduction to Kubeflow](https://youtu.be/cTZArDgbIWw)

[● Orchestrating TFX Pipelines](https://www.tensorflow.org/tfx/guide/kubeflow#kubeflow_pipelines)

[● Introduction to Machine Learning Pipelines with Kubeflow](https://rancher.com/blog/2020/introduction-to-machine-learning-pipeline)

[● Kubeflow — a machine learning toolkit for Kubernetes](https://medium.com/@michal.brys/kubeflow-a-machine-learning-toolkit-for-kubernetes-d8686f6c91b6)

[● ML for Mobile and Edge Devices - TensorFlow Lite](https://www.tensorflow.org/lite)

[● TensorFlow Lite Examples | Machine Learning Mobile Apps](https://www.tensorflow.org/lite/examples)

[● Optimize TensorFlow models for mobile and embedded devices](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/optimizing-neural-networks-for-mobile-and-embedded-devices-with-tensorflow)

[● The Essential Guide To Learn TensorFlow Mobile and Tensorflow Lite](https://towardsdatascience.com/the-essential-guide-to-learn-tensorflow-mobile-and-tensorflow-lite-a70591687800)