---
title: "First ML model"
---
<html>
    <body>
        <h2>Training Data</h2>
        <p>Input: &nbsp; [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]</p>
        <p>Output: [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0]</p>
        <!-- Take epochs from user and give train button -->
        <label for="epochs_input">Epochs to train the model:</label>
        <input type="number" id="epochs_input" name="epochs_input" min="1" width="20px">
        <button type="button" id="train_button" onclick="trainModel()">Train Model</button><br>
        <p id="message">Untrained Model</p>
        <!-- take user input and test model -->
        <label for="test_input">Number to test the model:</label>
        <input type="number" id="test_input" name="test">
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