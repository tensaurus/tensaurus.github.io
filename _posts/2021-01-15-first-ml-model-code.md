---
title: "Code Description for First ML Model"
author_profile: false
---

So you want to know how the [First Machine Learning Model]({% post_url 2020-11-19-first-ml-model %}) works under the hood. Whole thing is working in the browser itself. We have used a *Deep Learning* framework **Tensoflow.js** for the machine learning stuff. It is a library for machine learning in JavaScript.
{: style="text-align: justify;"}

{% include_relative  html/first-ml-model.html %}

## Code

<script src="https://gist.github.com/tensaurus/875489629e84aab566432692508fb0c2.js"></script>

## Code Explanation

Lines 4 to 13 are just the `html` code for the UI for the [First Machine Learning Model]({% post_url 2020-11-19-first-ml-model %}). First the user can provide number of `epochs` to `train` the `model`. Then train the model by clicking **Train Model** button. After training is done model can be tested for any given input by clicking **Test Model** button. The result is shown next to **Model Prediction** text. [What is a Model ?]({% post_url 2020-11-19-first-ml-model %}#what-is-a-model-) and [What is meant by Training a Model ?]({% post_url 2020-11-19-first-ml-model %}#what-is-meant-by-training-a-model-) are already answered. The term remain unknown is `epochs`.

### What is an Epoch ?

Number of Epochs is how many times we fit a model on the training data. An `epoch` is one round of training a model on whole training data. Model training is an iterative process. A model is trained for one epoch, loss is calculated by comparing model prediction and desired results, model weights are updated, model is trained for one more epoch and the cycle repeats untill get satisfactory results or we give give up hope on our model and decide to tweek the model.