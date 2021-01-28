---
title: "First Machine Learning Model"
# author_profile: false
---
{% include_relative  html/first-ml-model.html %}

Give any number to test the model and click the "Test Model" button. What is the Model Prediction? It can be any random number or if you are lucky it is double of what you have entered.

Now click **Train Model** and then test the model. Did you get double? If not again train the model. This time you must have got double of your input number. If still not then train more.

What's happening here? We have a set of inputs and corresponding outputs. You can easily figure out the relationship between then. Output is twice the input.

### Training Data

|Input:|-1.0|0.0|1.0|2.0|3.0|4.0|
|Output:|-2.0|0.0|2.0|4.0|6.0|8.0|

    y = 2x

We did not give the computer the straight forward formula for relationship between input and ouput. Rather, the computer is given set of inputs and known outputs. It is programmed to find the relationship between them by a **Machine Learning** technique called *Stochastic Gradient Descent*

## What is **Machine Learning** ?

Untill lately humans figured out the relation between inputs and outputs. We gave inputs and a mathematical function to computing machines. Then those machines gave outputs based on that information. This is what programming a computer meant.

Machine learning is a different approach to programming. In it computers are coded to figure out the relationship between the known set of inputs and outputs. Then we can ask the computer to give us the result of an input for which output is not known.

## What is a *Model* ?

In very simple terms a *Model* is a function which relates inputs to ouputs. We give it inputs and it gives ouputs.

## What is meant by *Training a Model* ?

A machine learning *Model* has a number of variable parameters called *weights*. Initially weights can have any value. To get the values of weights such that model can *predict* the best possible ouput for a given input, the model is trained. Training a model is iteratively updating its weights based on the difference between outputs computed by the model and the desired outputs for the given inputs. The difference between computed outputs and desired outputs is quantified as `loss`. There are different ways to calculate loss and update weights.

The model is trained till we get satifactory accuracy.When the model is trained, we can use it to predict results for inputs for which outputs are not known.

What is going inside the model, how it is defined, how it is trained and which machine learning framework is used to code it. If you are eager to know, dig in the source code of this page and look for model. To learn more go on reading the [code description]({% post_url 2021-01-15-first-ml-model-code %}).