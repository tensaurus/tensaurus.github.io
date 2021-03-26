---
title: "How to make Penguins Species Classifier in browser"
excerpt: "Make penguins species classifier in the browser using JavaScript deep learning library tensorflow.js. No need to install any software or package. All works in your browser."
toc: true
toc_sticky: true
---
# The Penguins Classifier App
As seen in the [Penguins Species Classifier]({% post_url 2021-01-08-penguins-classifier %}), once the model is trained, penguins can be classified into their species (Adelie, Chinstrap and Gentoo) for given set of Bill Length, Bill Depth and Flipper Length with a very good accuracy.
# The data
To make any machine learning model work we need relevant data. The penguins' data were collected and made available by [Dr. Kristen Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php) and the [Palmer Station, Antarctica LTER](https://pal.lternet.edu/), a member of the [Long Term Ecological Research Network](https://lternet.edu/). The [penguins data](https://github.com/allisonhorst/palmerpenguins) as R Package is made available by [Allison Horst](https://github.com/allisonhorst).

For this app we will be using [penguins data in CSV format](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv) provided by [Michael Waskom](https://github.com/mwaskom) because it is easier to work with CSV in tensorflow.js
## Sample Data

|species  |island   |bill_length_mm|bill_depth_mm|flipper_length_mm|body_mass_g|sex   |
|---------|---------|----------------|---------------|-----------------|-----------|------|
|Adelie   |Biscoe   |35.7            |16.9           |185.0            |3150.0     |FEMALE|
|Adelie   |Biscoe   |41.3            |21.1           |195.0            |4400.0     |MALE  |
|Gentoo   |Biscoe   |46.2            |14.4           |214.0            |4650.0     |MALE  |
|Gentoo   |Biscoe   |49.5            |16.2           |229.0            |5800.0     |MALE  |
|Chinstrap|Dream    |51.9            |19.5           |206.0            |3950.0     |MALE  |
|Chinstrap|Dream    |45.7            |17.0           |195.0            |3650.0     |FEMALE|

The dataset contains data for 344 penguins of three different species from three islands in the Palmer Archipelago, Antarctica.
To know more about the palmer penguins and their different features read [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/)

## Get the data
The tensorflow CSV Dataset is created by reading the [CSV file](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv) provided by Michael Waskom.
```js
const penguinsgUrl = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv';
const penguinsCSVData = tf.data.csv(penguinsgUrl, { mode: "no-cors" });
```
It is convenient to work in JavaScript arrays with tabular data. So we get the data from CSV Dataset into an array using the `toArray()` method of tf.data.Dataset class.
```js
const penguinsArrayData = await penguinsCSVData.toArray();
```
## Clean the data
The dataset has few null values. The rows with null values have to be removed. Take rows with all non-null values.
```js
const penguins = penguinsArrayData.filter(p => Object.values(p).every(e => e != null));
```
## Visualize data
As seen in the sample data, the dataset has seven variables, namely:

"species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"

In this machine learning task we want to classify the penguins according to species. So, species is our label and remaining variables are the features.

The pair plot below shows how the different pairs of features separate species of penguins.

{% include image-center.html src="penguins-pair-plot.png" alt="Pair plot for features of Palmer Penguins" caption="Pair plot for features of Palmer Penguins" %}

This pair plot was drawn using the python **seaborn** library in google colab notebook. To purely stick to our motive of doing this machine learning task in JavaScript we will use [tfjs-vis](https://js.tensorflow.org/api_vis/1.5.0/) library to plot and visualize penguins' features.

>tfjs-vis is a small library for in-browser visualization intended for use with TensorFlow.js.
--<cite>tfjs</cite>

The [`tfvis.render.scatterplot`](https://js.tensorflow.org/api_vis/1.5.0/#render.scatterplot) function renders a scatter plot of series of data. Here we have three series, one for each species of penguins, *Adelie*, *Chinstrap* and *Gentoo*.

```js
// map function for bill length and bill depth
const billLengthVSBillDepth = (p) => ({ x: p.bill_length_mm, y: p.bill_depth_mm });
const adelie = penguins.filter(p => p.species == "Adelie").map(billLengthVSBillDepth);
const chinstrap = penguins.filter(p => p.species == "Chinstrap").map(billLengthVSBillDepth);
const gentoo = penguins.filter(p => p.species == "Gentoo").map(billLengthVSBillDepth);
const plotElement = document.getElementById("plot");
tfvis.render.scatterplot(
    plotElement,
    { values: [adelie, chinstrap, gentoo], series: ['Adelie', 'Chinstrap', 'Gentoo'] },
    {
        zoomToFit: true,
        xLabel: 'Bill Length',
        yLabel: 'Bill Depth',
    }
);
```
The code above plots Bill Length vs Bill Depth. Similarly Bill Lenght vs Flipper Length can be plotted.

{% include image-half.html src-1="bill-length-vs-bill-depth.png" alt-1="Scatter plot for bill length vs bill depth" src-2="bill-length-vs-flipper-length.png" alt-2="Scatter plot for bill length vs flipper length" caption="Bill Length vs Bill Depth and Bill Length vs Flipper Length" %}

## Feature Selection
From the plots we can say the *Bill Length*, *Bill Depth* and *Flipper Length* separates the penguins in species satisfactorily. Thus we can use these features to train a model to classify penguins species.
## Preprocess the data

### Remove null values

### Training-Validation Split
To monitor the real progress of the model training it should be validated for the data which is not used for training. Before starting to train the model a portion of data is taken off from the dataset as validation data. For training a classification model we randomly take validation data.

To randomly select elements from a JavaScript array the best way is to first shuffle the array and then split the array in desired proportions using `slice` method of JavaScript array class.

```js
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
// make a copy of penguins
const penguinsShuffle = penguins.slice(0);
// shuffle it in-place
shuffle(penguinsShuffle);
```
The `shuffle(array)` function shuffles a given `array` inplace using  [Fisher-Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle).
Make a copy of penguins array, in-case we may need it later. Shuffle the copy.
Then split penguins data in train and validation sets. Here 30% of data is kept for validation set.
```js
const penguinsTrain = penguinsShuffle.slice(0, Math.floor(0.7 * penguinsShuffle.length));
const penguinsValid = penguinsShuffle.slice(Math.floor(0.7 * penguinsShuffle.length));
```
## Prepare features and label
### Features of Inputs
The elements of penguins data array are JavaScript objects with key-value pairs of seven variables. One such element is shown below.
```json
{
  "species": "Gentoo",
  "island": "Biscoe",
  "bill_length_mm": 46.2,
  "bill_depth_mm": 14.1,
  "flipper_length_mm": 217,
  "body_mass_g": 4375,
  "sex": "FEMALE"
}
```
But for the purpose of training a model in tfjs we just need the values. For inputs we will use values of variables *bill length*, *bill depth* and *flipper length*.
```js
const inputsTrain = penguinsTrain.map(p => ([
    p.bill_length_mm,
    p.bill_depth_mm,
    p.flipper_length_mm,
    // p.body_mass_g,
    // p.sex == 'MALE' ? 1 : 0
    ]));
```
`inputsTrain` is the inputs array, made by mapping values of desired features. Other feature values are commented out, you may try with different features if you wish to experiment. Similarly `inputsValid` can be prepared for validation input data.
### Label or Output
The variable *species* is the model label or output. The penguins data has three different string values for species, it can be *Adelie*, *Chinstrap* or *Gentoo*. But to use in the tfjs model numerical values for labels are required. One way to achieve it is *One Hot Encoding* the species variable.
### One Hot Encode Labels
The species variable is represented as an array of `0`s and `1`s. The species variable can have three different values, so, the label array will have three elements which can be `0` or `1` depending on the type of species. Each element of the label array represents a specie. For any given label the element associated with the specie is `1` and rest are `0`. For example, if the first element is for *Adelie*, second for *Chinstrap* and third for *Gentoo* then for `"species": "Gentoo"` the label will be `[0, 0, 1]`. This is done by mapping species values to `1` or `0` according to the type of species.
```js
const labelsTrain = penguinsTrain.map(p => ([
    p.species == 'Adelie' ? 1 : 0,
    p.species == 'Chinstrap' ? 1 : 0,
    p.species == 'Gentoo' ? 1 : 0
    ]));
```
Till now the inputs and outputs are in JavaScript arrays. Before moving forward they have to be converted to `tf.tensors`. [Why and how the data is converted to tfjs tensors?]({% post_url 2021-01-15-first-ml-model-code %}#prepare-data)
```js
const inputTensorTrain = tf.tensor2d(inputsTrain, [inputsTrain.length, numOfFeatures]);
const labelTensorTrain = tf.tensor2d(labelsTrain, [labelsTrain.length, 3]);
```
It is faster and easier to manipulate tensorflow tensors as we can take benefit of vectorization and broadcasting in doing array operations. You can see this in action while normalizing the data.

### Normalize the data
If the values of input and output variables are between 0 and 1 the training of a Machine Learning model starts with smaller loss, converges faster and has minimal effect of random initialization. As you have seen in [Code Description for Celsius-Fahrenheit ML Model]({% post_url 2021-02-03-code-celsius-fahrenheit %}#controlled-randomness) the random initialization is controlled by fixing the random seed value.

The penguins data variables has values with different ranges and magnitude. For efficient model training the data needs to be normalized before feeding it into the model for training. It is very simple to normalize a data series, just subtract the minimum value from it and divide the resultant by the range (difference of maximum and minimum values) of the data series.

Normalization is easily done with tfjs tensors
```js
inputMax = inputTensorTrain.max(0);
inputMin = inputTensorTrain.min(0);
const normalizedInputsTrain = inputTensorTrain.sub(inputMin).div(inputMax.sub(inputMin);
```
# Define Model
Based on the machine learning task at hand, the form of input data and ouputs a suitable neural network is defined. Which includes Model Architecture, Loss function, Optimizer and other Hyperparameters.
## Model Architechture
For the penguins species classification task at hand a sequential model with two layers would suffice. [*Brief explanation of defining a simple neural network model*]({% post_url 2021-01-15-first-ml-model-code %}). For penguins classifier the model is defined as below.
```js
model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [numOfFeatures], units: 3 }));
model.add(tf.layers.dense({ activation: "softmax", units: 3 }));
model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.05),
    metrics: ['acc']
});
```
Earlier in Celsius to Fahrenheit ML model we have used just one layer. Here at least two layers are required. First is the input layer and second is the output layer. In second dense layer number of units is three because penguins are to categorized in to three classes and one new term `activation: "softmax"` is used. An activation function transforms the output of a neural network layer. What are activation functions and how to choose one, explained nicely [here](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/) by Jason Brownlee of machinelearninmastery. For a multi-class classification task the [softmax activation function](https://machinelearningmastery.com/softmax-activation-function-with-python/) is used.
## Loss and Optimizer
The loss fuction used for this model is `categoricalCrossentropy`, the name itself is self explanatory. [The adam optimizer]({% post_url 2021-02-03-code-celsius-fahrenheit %}#adam-ance-optimizer) is used here with learning rate of 0.05 to speed up the training. And, we track accuracy metrics during the training process. Other [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) are selected in the model training function `model.fit`.
# Train the Model
The code for training the neural network model is as below,
```js
await model.fit(normalizedInputsTrain, labelTensorTrain, {
        batchSize: 32,
        epochs: numOfEpochs,
        shuffle: true,
        validationData: [normalizedInputsValid, labelTensorValid],
        callbacks: [tfvis.show.fitCallbacks(
            fitElement,
            ['loss', 'acc'],
            {height: 200, seriesColors: ["BlueViolet"], callbacks: ['onEpochEnd'] }
        ),
        tfvis.show.fitCallbacks(
            fitElement1,
            ['val_loss', 'val_acc'],
            {height: 200, seriesColors: ["GreenYellow"], callbacks: ['onEpochEnd'] }
        ),
        {
            onEpochEnd: async (epoch, logs) => {
                const bar = document.getElementById("myBar");
                bar.style.width = (epoch + 1) * 100 / numOfEpochs + "%";
                bar.innerHTML = epoch + 1;
                console.log("Epoch: " + epoch
                    + " Loss: " + logs.loss.toFixed(4)
                    + " Accuracy: " + logs.acc.toFixed(4)
                    + " Val Loss: " + logs.val_loss.toFixed(4)
                    + " Val Accuracy: " + logs.val_acc.toFixed(4));
            }
        }]
    });
}
```
Let's go through the arguments passed to the `model.fit` function one by one.
1. The first one `normalizedInputsTrain` is the normalized input data
1. The Second argument `labelTensorTrain` is the one hot encoded labels for penguin species.
1. The third argument is an object containing different optional fields.
    1. Your are familier with `batchSize`
    1. and `epochs`.
    1. The `shuffle` field takes boolean value, if its `true`, the training data is shuffled before each epoch. So that batches have different composition of training data for better model fitting.
    1. with `validationData: [normalizedInputsValid, labelTensorValid]` the validation data is given to the model during training and model performance on it is evaluated after each epoch.
    1. then there is `callbacks` field which takes a list of callbacks which are called during training. Here we have used three callbacks, first two are from tfjs-vis library to graphically display  loss and accuracy for training and validation which are updated at each epoch end. The third callback `onEpochEnd` is called at each epoch end and can use current epoch number and logs. Here it is used to update the progress bar to display training progress and to show different metrics in console.
The model is fit for 20 epochs and good level of training (>98%) and validation (>95%) accuracy is achieved. The results may differ ever slightly on each page reload but the accuracy obtained each time will be very good. No need to worry about random initialization.
# Test the Model
You can test the model in the [browser page]({% post_url 2021-01-08-penguins-classifier %}) itself by entering values for features of a penguin. Click on the [**Classify Penguin**]({% post_url 2021-01-08-penguins-classifier %}#penguins-classifier-app){: .btn .btn--primary} button to know the species to which this penguin belongs.
# Deploy the Model
Although the model deployment is done in [the same browser page]({% post_url 2021-01-08-penguins-classifier %}) where all the machine learning steps are carried out. That is the beauty of tensoflow.js - all the machine learning in the browser itself. But, to separate the training and deployment we can train and save the model and then use it with UI just sufficient for inference. More on model deployment in JavaScript using tfjs in later post.
