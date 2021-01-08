<html>
<head></head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    <script lang="js">
        let numOfFeatures;
        let convertedTrainingData;
        let convertedTestingData;
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
            const model = tf.sequential();
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
            // Test Cases:
            
            // Adelie
            // const testVal = tf.tensor2d([41, 21], [1, 2]);

            // Chinstrap
            // const testVal = tf.tensor2d([50.8, 19], [1, 2]);

            // Gentoo
            const testVal = tf.tensor2d([45, 14], [1, 2]);
                   
            
            const prediction = model.predict(testVal);
            const pIndex = tf.argMax(prediction, axis=1).dataSync();
            
            const classNames = ["Adelie", "Chinstrap", "Gentoo"];
            
            // alert(prediction)
            alert(classNames[pIndex])
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
        trainModel();
        </script>
<body>
    <!-- <h1>Simple Tabular Classifier: Iris Flower</h1> -->
</body>
</html>