---
title: "First Machine Learning Model"
author_profile: false
---
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

A machine learning *Model* has a number of variable parameters called *weights*. Initially weights can have any value. To get the values of weights such that model can *predict* the best possible ouput for a given input, the model is trained. Training a model is iteratively updating its weights based on the difference between outputs computed by the model and the desired outputs for the given inputs.

The model is trained till we get satifactory accuracy.When the model is trained, we can use it to predict results for inputs for which outputs are not known.

What is going inside the model, how it is defined, how it is trained and which machine learning framework is used to code it. If you are eager to know, dig in the source code of this page and look for model. To learn more go on reading the [code description]({% post_url 2021-01-15-first-ml-model-code %}).