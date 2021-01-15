---
title: "Second ML model"
---
## Training Data

|Fahrenheit:|86.0|95.0|98.6|100.4|104.0|113.0|
|Celsius:|30.0|35.0|37.0|38.0|40.0|45.0|

<html>
    <body>
        <span>Epochs to train the model:</span>
        <input type="number" id="epochs_input" min="1" value="100" style="width: 4em;">
        <button type="button" id="train_button" onclick="trainModel()">Train Model</button><br>
        <p id="message">Untrained Model</p>
        <!-- take user input and test model -->
        <span>Temperature: </span>
        <input type="number" id="test_input" value="98.6" style="width: 4em;"><span>&deg;F</span>
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
        const x = tf.tensor2d([86.0, 95.0, 98.6, 100.4, 104.0, 113.0], [6, 1]);
        const y = tf.tensor2d([30.0, 35.0, 37.0, 38.0, 40.0, 45.0], [6, 1]);
        let user_epochs;
        // First we define a very simple machine learning model.
        const model = tf.sequential();
        model.add(tf.layers.dense({units: 1, inputShape: [1]}));
        // Model training didn't converge with 'sgd' optimizer. Let's try 'adam' optimizer
        // Train for 100 epochs. Loss less than 5 will do. Try for 100 more epochs few times
        // till you get loss<5. If loss is still high, just reload the page and try again.
        // Reloading works because model is initialized with random parameters.
        model.compile({loss:'meanSquaredError', 
                       optimizer: 'adam'});
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
            const train_btn = document.getElementById("train_button");
            const train_msg = document.getElementById("message");
            test_btn.disabled = true;
            train_btn.disabled = true;
            train_msg.innerHTML = 'Hold on!! Model training';
            doTraining(model).then(() => {
            test_btn.disabled = false;
            train_btn.disabled = false;
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
            document.getElementById("result").innerHTML = 'Model Prediction: ' + output_number + '&deg;C';
        }
    </script>
</html>

Enter any temperature in degrees Fahrenheit and click "Test Model" button. You probably won't get the expected output, temperature in degrees Celsius. 