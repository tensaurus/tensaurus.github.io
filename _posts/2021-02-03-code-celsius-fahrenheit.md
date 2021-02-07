---
title: "Code Description for Celsius-Fahrenheit ML Model"
author_profile: false
toc: true
toc_sticky: true
excerpt: "For implementation of this model also **TFJS** is used. There are few changes to the code which were essential to make it work easily. I will elaborate mainly on those changes in this post. For detailed explanation go through the [Code Description for First ML Model]({% post_url 2021-01-15-first-ml-model-code %})."
---
## The Data

|**Training Data**
|Fahrenheit:|86.0|95.0|98.6|100.4|104.0|113.0|
|Celsius:|30.0|35.0|37.0|38.0|40.0|45.0|

## The Model UI

{% include_relative html/ml-celsius-fahrenheit.html %}

## The Code

<script src="https://gist.github.com/tensaurus/8489acee17629a930c9e89cccd465ce2.js"></script>

Most of the code for [Celsius-Fahrenheit Machine Learning Model]({% post_url 2020-11-20-ml-celsius-fahrenheit %}) is similar to the [code for first ML model]({% post_url 2021-01-15-first-ml-model-code %}). For implementation of this model also **TFJS** is used. There are few changes to the code which were essential to make it work easily. I will elaborate mainly on those changes in this post. For detailed explanation go through the [Code Description for First ML Model]({% post_url 2021-01-15-first-ml-model-code %}).

## Controlled Randomness

Converting of training data to tensors and model initialization takes place on each page load. When the model is initialized its parameters, weights and bias for each layer, get random initial values. Through the course of training these values get updated using some optimizer function.

```js
// Fix seed for random initialization
Math.seedrandom(13);
// Initialize a Sequential model
const model = tf.sequential();
// Add layers to the model
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
```
Before defining the model the random seed is fixed by using `Math.seedrandom(13)`. This ensures that the model parameters are same each time the model is initialized. This way model training will proceed the same way and gives same end result each time.

If the random seed is not fixed, then after each page load, during training loss starts with different value and varies in different way and gives different result each time. You can see this while training [the first model]({% post_url 2020-11-19-first-ml-model %}) by looking at the loss values in JS console. Train the model, see the loss values for first few epochs, now reload the page and again train the model. This time you will find different loss values. Each time you will get different values. The values may repeat if by chance same radom initialization occured.

You will not find such behaviour for the [Celsius Fahrenheit model]({% post_url 2020-11-20-ml-celsius-fahrenheit %}) as the seed for random number generator is fixed here to 13. You may wonder why 13? why not any other number. I have tried different seeds from 1 onwards and found minimum initial loss and best convergence for seed 13. You are welcome to try and find some better value for the random seed.

## Adam-*ance* Optimizer

```js
model.compile({loss:'meanSquaredError', 
                optimizer: 'adam'});
```
In this case the optimizer used for updating model parameters is `adam`, whereas, you can see in the [code for first ML model](({% post_url 2021-01-15-first-ml-model-code %})) the `sgd` optimizer was used. The job of the `adam` optimizer is same as `sgd` - to update the model parameters using the loss computed after each batch of training - but based on a different algorithm. The Adam Optimizer uses the [Adam gradient descent algorithm](https://arxiv.org/abs/1412.6980).

Why the `adam` optimizer was used to train the model on this dataset? The simple answer is `sgd` optimizer was not able to train the model so I tried `adam` and it worked. The loss values were very high with `sgd` and the loss was not converging. With Adam optimizer and random seed as 13 the loss could be converged to acceptable value in 100 epochs. The Adam algorithm has now become the go-to optimizer for most machine learning models.

One important lesson a machine learning practitioner learns is that it is part logic-and-analysis, and part, hit-and-trial.

Rest of the code is same as [the code for first ML model](({% post_url 2021-01-15-first-ml-model-code %})).