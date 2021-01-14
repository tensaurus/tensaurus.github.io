<html>
<head></head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    <script lang="js">
        let numOfFeatures;
        let convertedTrainingData;
        let convertedTestingData;
        const model = tf.sequential();
        function prepareData(){
            let repeat = true;
            const trainingUrl = '/assets/csv/penguins_train.csv';
            const trainingData = tf.data.csv(trainingUrl, {
                columnConfigs: {
                    species: {
                        isLabel: true
                    },
                    culmen_length_mm: {
                        required: true
                    },
                    culmen_depth_mm: {
                        required: true
                    // },
                    // flipper_length_mm: {
                    //     required: true
                    // },
                    // body_mass_g: {
                    //     required: true
                    }
                    // taken only two features, viz, culmen_length_mm and culmen_depth_mm as they
                    // seggregates the dataset in species in best way as per pairplot
                },
                configuredColumnsOnly: true
            });
            // trainingData.take(1).forEachAsync(e => console.log(e));
            // numOfFeatures = (await trainingData.columnNames()).length - 1;
            numOfFeatures = 2;
            // console.log(numOfFeatures);
            // const numOfSamples = 150;
            convertedTrainingData =
                  trainingData.map(({xs, ys}) => {
                      const labels = [
                            ys.species == "Adelie" ? 1 : 0,
                            ys.species == "Chinstrap" ? 1 : 0,
                            ys.species == "Gentoo" ? 1 : 0
                      ] 
                    //   labels is already an array, Object.values not reqd for it
                      return{ xs: Object.values(xs), ys: labels};
                //   });
                  }).batch(18);
            // convertedData.take(10).forEachAsync(e => console.log(e));
            const testingUrl = '/assets/csv/penguins_test.csv';
            const testingData = tf.data.csv(testingUrl, {
                columnConfigs: {
                    species: {
                        isLabel: true
                    },
                    culmen_length_mm: {
                        required: true
                    },
                    culmen_depth_mm: {
                        required: true
                    // },
                    // flipper_length_mm: {
                    //     required: true
                    // },
                    // body_mass_g: {
                    //     required: true
                    }
                    // taken only two features, viz, culmen_length_mm and culmen_depth_mm as they
                    // seggregates the dataset in species in best way as per pairplot
                },
                configuredColumnsOnly: true
            });
            // trainingData.take(1).forEachAsync(e => console.log(e));
            // const numOfFeatures = (await trainingData.columnNames()).length - 1;
            // console.log(numOfFeatures);
            // const numOfSamples = 150;
            convertedTestingData =
                  testingData.map(({xs, ys}) => {
                      const labels = [
                            ys.species == "Adelie" ? 1 : 0,
                            ys.species == "Chinstrap" ? 1 : 0,
                            ys.species == "Gentoo" ? 1 : 0
                      ] 
                    //   labels is already an array, Object.values not reqd for it
                      return{ xs: Object.values(xs), ys: labels};
                //   });
                  }).batch(8);
            // convertedTestingData.take(24).forEachAsync(e => console.log(e));
                }
        async function trainModel(){
            // Math.seedrandom(3);
            model.add(tf.layers.dense({inputShape: [numOfFeatures], activation: "relu", units: 16}));
            model.add(tf.layers.dense({activation: "relu", units: 16}))
            model.add(tf.layers.dense({activation: "relu", units: 8}))
            model.add(tf.layers.dense({activation: "softmax", units: 3}));
            model.compile({loss: "categoricalCrossentropy", optimizer: 'adam', metrics: ['acc']});
            let acc;
            const numOfEpochs = 100;
            await model.fitDataset(convertedTrainingData, 
                             {epochs: numOfEpochs,
                                validationData: convertedTestingData,
                              callbacks:[
                                  new tf.CustomCallback({
                                    onEpochEnd: async(epoch, logs) =>{
                                        // acc = logs.acc;
                                        const bar = document.getElementById("myBar");
                                        bar.style.width = epoch*100/numOfEpochs + 1 + "%";
                                        bar.innerHTML = epoch;
                                        if (epoch == numOfEpochs - 1) {
                                            console.log("Epoch: " + epoch 
                                                  + " Loss: " + logs.loss.toFixed(4) 
                                                  + " Accuracy: " + logs.acc.toFixed(4) 
                                                  + " Val Loss: " + logs.val_loss.toFixed(4) 
                                                  + " Val Accuracy: " + logs.val_acc.toFixed(4));
                                        }
                                    },
                                    // onTrainEnd: async(logs) =>{
                                    //     console.log(Object.keys(logs));
                                        // if (acc>0.5) {
                                        //     repeat = false;
                                        //     console.log(repeat);
                                        // }
                                    // }
                                    }),
                                // tf.callbacks.earlyStopping({monitor: 'acc', patience: 20})
                            ]});
            // return repeat;
        }
        // Run the function run after the page is loaded.
        // document.addEventListener('DOMContentLoaded', run);
        // run (initialize and train model) for few times using while till desired acc achieved
        // let train = true;
        // (async ()=>{
        //     while (train) {
        //         train = await run();
        //     }
        // })();
        // run();
        prepareData();
        // (async ()=>{
        //     for (let i = 0; i < 10; i++) {
        //         Math.seedrandom(i);
        //         console.log('seed = ', i);
        //         await trainModel();
        //     }
        // })();
        // trainModel();
        function train() {
            // user_epochs = Number(document.getElementById("epochs_input").value);
            const test_btn = document.getElementById("test_button");
            const train_btn = document.getElementById("train_button");
            const train_msg = document.getElementById("message");
            test_btn.disabled = true;
            train_btn.disabled = true;
            train_msg.innerHTML = 'Hold on!! Model training';
            trainModel().then(() => {
                test_btn.disabled = false;
                train_btn.disabled = false;
                train_msg.innerHTML = 'Model Trained!! Now test the model';
            });
        }
        function testModel() {
            // Get the user input value and convert it to number
            const input_1 = Number(document.getElementById("cul_len").value);
            const input_2 = Number(document.getElementById("cul_dep").value);
            // const flip_len = Number(document.getElementById("input_3").value);
            // const bod_mas = Number(document.getElementById("input_4").value);
            // Get prediction from the model for user given input. Model prediction
            // is a Tensor
            const prediction = model.predict(tf.tensor2d([input_1, input_2], [1,2]));
            const pIndex = tf.argMax(prediction, axis=1).dataSync();
            const classNames = ["Adelie", "Chinstrap", "Gentoo"];
            // alert(prediction)
            alert(classNames[pIndex])
            // Get the numerical value from the tensor using dataSync() and round it
            // document.getElementById("result").innerHTML = 'Model Prediction: ' + output_number;
        }
        </script>
<body>
    <!-- <h1>Simple Tabular Classifier: Iris Flower</h1> -->
    <div id="myProgress" style="width: 100%; background-color: #ddd;">
        <div id="myBar" style="width: 1%; height: 30px; background-color: #4CAF50; text-align: center; ;line-height: 32px; color: black;">
        </div>
    </div>
    <p id="message">Untrained Model</p>
    <button type="button" id="train_button" onclick="train()">Train Model</button><br>
    <span>Culmen Length:</span><input type="number" id="cul_len" style="width: 4em;"><br>
    <span>Culmen Depth:</span><input type="number" id="cul_dep" style="width: 4em;"><br>
    <button type="button" id="test_button" onclick="testModel()">Test Model</button><br>
</body>
</html>