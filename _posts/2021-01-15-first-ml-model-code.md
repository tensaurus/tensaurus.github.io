---
title: "Code Description for First ML Model"
author_profile: false
---

So you want to know how the [First Machine Learning Model]({% post_url 2020-11-19-first-ml-model %}) works under the hood. Whole thing is working in the browser itself. We have used a *Deep Learning* framework **Tensoflow.js** for the machine learning stuff. It is a library for machine learning in JavaScript.
{: style="text-align: justify;"}

*~*~*~*~ reference coe=de from somewhere

<html>
    <body>
        <span>Epochs to train the model:</span><input type="number" id="epochs_input" min="1"
        value="10" style="width: 4em;">
        <button type="button" id="train_button" onclick="trainModel()">Train Model</button><br>
        <p id="message">Untrained Model</p>
        <!-- take user input and test model -->
        <span>Number to test the model:</span><input type="number" id="test_input"
        value="2" style="width: 4em;">
        <button type="button" id="test_button" onclick="testModel()">Test Model</button>
        <br>
        <p id="result">Model Prediction: </p>
    </body>
    <!-- To have Tensorflow in your browser add the right source in script tag -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    <script lang="js">
        /*
        x are the inputs and y are the outputs. For this set of inputs, outputs are known 
        but the relationship between them is not known. In traditional computing we design 
        the system to get the desired output. But in machine learning we let the system learn
        the relationship by going through the epochs of training.
        */
       /*
       For input "1.0" output is "2.0" and for input "2.0" the output is "4.0".
       You can easily guess the mathematical relationship betweent the two
       set of numbers. It is "y = 2x". For this simple case we could guess it.
        */
        // In tensorflow.js they are defined as follows
        const x = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
        const y = tf.tensor2d([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0], [6, 1]);
        let user_epochs;
        // First we define a very simple machine learning model.
        const model = tf.sequential();
        model.add(tf.layers.dense({units: 1, inputShape: [1]}));
        model.compile({loss:'meanSquaredError', 
                       optimizer:'sgd'});
        /*
        function to train the model with given set of inputs and outputs for number of epochs
        as specified and a callback to show loss after each epoch in the console
        */
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
        // Finally train the model by calling the function "doTraining"
        function trainModel() {
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
        // function to test the model
        function testModel() {
            // Get the user input value and convert it to number
            const input = Number(document.getElementById("test_input").value);
            // Get prediction from the model for user given input. Model prediction
            // is a Tensor
            const output = model.predict(tf.tensor2d([input], [1,1]));
            // Get the numerical value from the tensor using dataSync() and round it
            const output_number = output.dataSync()[0].toFixed(0);
            document.getElementById("result").innerHTML = 'Model Prediction: ' + output_number;
        }
    </script>
</html>

<script src="https://gist.github.com/tensaurus/875489629e84aab566432692508fb0c2.js"></script>

## Code Explanation

Lines 4 to 13 are just the `html` code for the UI for the [First Machine Learning Model]({% post_url 2020-11-19-first-ml-model %}). First the user can provide number of `epochs` to `train` the `model`. Then train the model by clicking **Train Model** button. After training is done model can be tested for any given input by clicking **Test Model** button. The result is shown next to **Model Prediction** text. [What is a Model ?]({% post_url 2020-11-19-first-ml-model %}#what-is-a-model-) and [What is meant by Training a Model ?]({% post_url 2020-11-19-first-ml-model %}#what-is-meant-by-training-a-model-) are already answered. The term remain unknown is `epochs`.

### What is an Epoch ?

Number of Epochs is how many times we fit a model on the training data. An `epoch` is one round of training a model on whole training data. Model training is an iterative process. A model is trained for one epoch, loss is calculated by comparing model prediction and desired results, model weights are updated, model is trained for one more epoch and the cycle repeats untill get satisfactory results or we give give up hope on our model and decide to tweek the model.