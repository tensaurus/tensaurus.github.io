---
title: "Second ML model"
excerpt: "This Machine Learning model is very similar to [the first model]({% post_url 2020-11-19-first-ml-model %}) you may have seen in the earlier [post]({% post_url 2020-11-19-first-ml-model %}). We want to train a model to convert temperature in degrees Fahrenheit to Celsius."
---
## Model UI

{% include_relative html/ml-celsius-fahrenheit.html %}

This Machine Learning model is very similar to [the first model]({% post_url 2020-11-19-first-ml-model %}) you may have seen in the earlier [post]({% post_url 2020-11-19-first-ml-model %}). There the model was expected to learn the relationship between two sets of numbers. This time the set of numbers are more meaningful. We want to train a model to convert temperature in degrees Fahrenheit to Celsius.

|**Training Data**
|Fahrenheit:|86.0|95.0|98.6|100.4|104.0|113.0|
|Celsius:|30.0|35.0|37.0|38.0|40.0|45.0|

Enter any temperature in degrees Fahrenheit and click [**Test Model**](#){: .btn .btn--primary} button. You won't get the expected temperature in degrees Celsius as the model is not trained yet. Click on [**Train Model**](#){: .btn .btn--primary} button. You can provide the number of epochs. The default value of 100 epochs trains the model just right.

Now test the model by clicking [**Test Model**](#model-ui){: .btn .btn--primary} button. You will get the temperature value very near to the expected value. The loss values are printed in console after each epoch end. You can have a look by opening JS console (`Ctrl`+`Shift`+`J` for Chrome). Loss decreases as training proceeds.

## What's under the hood

Although, with a little bit of mathematical manipulation you can deduce the relationship between temperature numbers in Celsius and Fahrenheit from the given data. This example is just an attempt to show *how to program computers to find relationship between the given inputs and outputs* or simply *Machine Learning*.

|Fahrenheit:|86.0|95.0|98.6|100.4|104.0|113.0|
|Celsius:|30.0|35.0|37.0|38.0|40.0|45.0|

In this example also, like [the first model]({% post_url 2020-11-19-first-ml-model %}), a single layered neural network model is used as the model architechture (a very heavy word for our very light model). This time few tricks were used for training the model on this data, which were necessary to make it work. If you are interested to know the details read the [code description for this Celsius-Fahrenheit Machine Learning Model]({% post_url 2021-02-03-code-celsius-fahrenheit %}).