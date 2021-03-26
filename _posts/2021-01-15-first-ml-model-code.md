---
title: "Code Description for First ML Model"
author_profile: false
toc: true
toc_sticky: true
---

So you want to know how the [First Machine Learning Model]({% post_url 2020-11-19-first-ml-model %}) works under the hood. Whole thing is working in the browser itself. We have used a *Deep Learning* framework **Tensoflow.js** for the machine learning stuff. It is a library for machine learning in JavaScript.
{: style="text-align: justify;"}

{% include_relative  html/first-ml-model.html %}

## Code

<script src="https://gist.github.com/tensaurus/875489629e84aab566432692508fb0c2.js"></script>

## Code Explanation

Lines 4 to 13 are just the `html` code for the UI for the [First Machine Learning Model]({% post_url 2020-11-19-first-ml-model %}). First the user can provide number of `epochs` to `train` the `model`. Then train the model by clicking [**Train Model**](#){: .btn .btn--primary} button. After training is done model can be tested for any given input by clicking [**Test Model**](#){: .btn .btn--primary} button. The result is shown next to *Model Prediction* text. [What is a Model ?]({% post_url 2020-11-19-first-ml-model %}#what-is-a-model-) and [What is meant by Training a Model ?]({% post_url 2020-11-19-first-ml-model %}#what-is-meant-by-training-a-model-) are already answered. The term remain unknown is `epochs`.

*What is an Epoch ?*

Number of Epochs is how many times we fit a model on the training data. An `epoch` is one round of training a model on whole training data. Model training is an iterative process.
- A model is trained for one epoch.
- `Loss` is calculated by comparing model prediction and desired results.
- Model `weights` (variable parameters) are updated.
- model is trained for one more epoch.

... and the cycle repeats untill get satisfactory results or we give give up hope on our model and decide to tweek the model.

**`loss`**: Numerical value representing difference between model prediction and training data outputs. There are different mathematical models to define loss, like, `meanSquaredError`, `huberLoss`, `logLoss`, etc. [Losses available in tensorflow.js](https://js.tensorflow.org/api/latest/#Training-Losses).
{: .notice--info}

*Code Explanation continued ...*

Now coming to JavaScript part. Line 16 to the last is all JavaScript. The machine learning terms, like `tf.tensor2d`, `tf.sequential`, `layers`, `compile`, `loss`, `optimizer` and many more, are referring to the machine learning library **Tensorflow.js**. It is a JavaScript library so the whole code works in browser.

*Get Tensorflow.js*

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
```
To get the **Tensorflow.js** library in the browser the script tag shown above is used. It gets the latest tensorflow.js from the CDN.

### Prepare data

To train the model it has to be provided with data to fit on. A Tensorflow.js model takes data in the form of [tf.tensors](https://js.tensorflow.org/api/latest/#class:Tensor).

<cite>A tf.Tensor object represents an immutable, multidimensional array of numbers that has a shape and a data type.</cite> --- [TFJS API](https://js.tensorflow.org/api/latest/)
{: .notice}

The input and output data is converted to tensors using `tensor2d()` function from tensorflow.js library, referenced as `tf`.

```js
    const x = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
    const y = tf.tensor2d([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0], [6, 1]);
```
The first argument in `tf.tensor2d()` `[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]` is the array of input data and the second argument `[6, 1]` defines the shape of the tensor. This is just converting the data in a format which tfjs (short for tensorflow.js) model understands. If you are interested you can know more about [tf.tensor2d()](https://js.tensorflow.org/api/latest/#tensor2d) in the tfjs documentation.

### Define the Model

Now we will construct the model in tensorflow.js. To fit this simple data a *sequential* neural network is used as our machine learning model. In a sequential neural network model the input data passes from one layer to another in the sequence as defined while constructing the model.

Model is initialized with `tf.sequential()`,

```js
const model = tf.sequential();
```
then desired number of layers are added using `model.add()`. As this is a single layered neural network model `model.add()` is used once,

```js
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
```
in the argument to `model.add()` we can define which [type of neural network layers](https://js.tensorflow.org/api/latest/#Layers) we want to use and different parameters related to that layer. Here a [`dense layer`](https://js.tensorflow.org/api/latest/#Layers) is added to the model. `inputShape: [1]` says that input at a time is a single value and `units: 1` declares the number of nodes the neural network layer should have.

<figure style="width: 400px" class="align-center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/A_simple_neural_network_with_two_input_units_and_one_output_unit.png" alt="A simple neural network with two input units and one output unit">
  <figcaption>A simple neural network<br><cite>Image Credit: <a href="https://commons.wikimedia.org/wiki/File:A_simple_neural_network_with_two_input_units_and_one_output_unit.png">AI456</a>, <a href="https://creativecommons.org/licenses/by-sa/3.0">CC BY-SA 3.0</a>, via Wikimedia Commons</cite></figcaption>
</figure> 

After constructing the model we have to define how model *weights* are updated during training. For this we have to decide which `loss` function and `optimizer` to use.

```js
model.compile({loss:'meanSquaredError', 
                optimizer:'sgd'});
```

`Weights` are updated using one of many available [optimizers](https://js.tensorflow.org/api/latest/#Training-Optimizers).

Here we have used `meanSquaredError` loss and `sgd` for optimization. SGD stands for [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

Formating training data in tensors and defining model in tensorflow.js terms is done. Now comes the real thing

### Train the Model

```js
async function doTraining(model){
    const history = 
            await model.fit(x, y, 
                { epochs: user_epochs,
                    callbacks:{
                        onEpochEnd: async(epoch, logs) =>{
                            console.log("Epoch:" 
                                        + epoch 
                                        + " Loss:" 
                                        + logs.loss);
                        }
                    }
                });
}
```
`model.fit` takes the training data in `x` (inputs) and `y` (outputs), trains the model for `epochs: user_epochs` and at the end of each epoch prints epoch number and loss in console using callback `onEpochEnd`. Callbacks are handy in getting information and triggering some action for different milestones during model training. In browser js console you can see loss decreasing after each epoch as training goes on.

```js
function trainModel() {
    // Get the user epochs value and convert it to number
    user_epochs = Number(document.getElementById("epochs_input").value);
    const test_btn = document.getElementById("test_button");
    const train_msg = document.getElementById("message");
    test_btn.disabled = true;
    train_msg.innerHTML = 'Hold on!! Model training';
    doTraining(model).then(() => {
        test_btn.disabled = false;
        train_msg.innerHTML = 'Model Trained!! Now test the model';
    });
}
```
`trainModel()` function is written to link the UI to the model training code.`testModel()` allows users to test the model for inputs provided by them.

### Test the Model

`testModel()` function allows users to test the model for inputs provided by them.

```js
function testModel() {
    // Get the user input value and convert it to number
    const input = Number(document.getElementById("test_input").value);
    // Get prediction from the model for user given input.
    const output = model.predict(tf.tensor2d([input], [1, 1]));
    // Model prediction is a Tensor. Get the numerical value from the tensor
    const output_number = output.dataSync()[0].toFixed(0);
    // Display the model prediction
    document.getElementById("result").innerHTML = 'Model Prediction: ' + output_number;
}
```
The `model.predict()` computes the model output for any given input. But the input should be a tensor with same shape as any input of training data on which model is trained. `tf.tensor2d([input], [1, 1])` is very similar to how the input training data was defined. Any tfjs tensor cannot be directly included in html, [`dataSync()`](https://js.tensorflow.org/api/latest/#tf.Tensor.dataSync) method of tfjs gets the numerical value from a tensor.

This was the complete process of a very simple example of a Machine Learning model using tensorflow.js. A single layered neural network is used as the machine learning model here. The whole thing is running in the browser by importing tfjs library from a CDN.

The process of applying machine learning technique to a problem can be divided in these major steps:
1. [Prepare data](#prepare-data)
1. [Define the Model](#define-the-model)
1. [Train the Model](#train-the-model)
1. [Test the Model](#test-the-model)