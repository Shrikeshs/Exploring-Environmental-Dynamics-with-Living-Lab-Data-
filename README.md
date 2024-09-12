# Exploring Environmental Dynamics with Living Lab Data :
This repository hold all information for the dissertation (Exploring Environmental Dynamics: Analyzing Variations and Trends in Air Quality Surrounding Leeds Using Living Lab Air Quality Sensor Data). The repo will track the progress and will hold all steps taken towards the completion of the dissertation. (TBC)


## Introduction :
Air pollution has become a critical environmental issue, with particulate matter (PM) being one of the most concerning pollutants due to its adverse health effects. PM2.5, referring to fine particulate matter with a diameter of less than 2.5 micrometers, poses significant health risks as it can penetrate deep into the lungs and bloodstream. The ability to predict PM2.5 levels accurately is crucial for mitigating its impact on public health and implementing effective environmental policies.

This dissertation focuses on predicting PM2.5 concentrations using machine learning techniques. By leveraging a dataset that captures various environmental and meteorological factors, this research aims to uncover the relationships between these features and PM2.5 levels, providing valuable insights into air quality trends in Leeds. The core objective is to apply machine learning models, specifically Long Short-Term Memory (LSTM) networks, to analyze time-series data and identify patterns that influence PM2.5 concentrations.

In addition to the prediction task, this dissertation explores the complexities of data processing, feature engineering, and model optimization. Through hyperparameter tuning and model evaluation, the study seeks to find the most effective approach for minimizing prediction error while maintaining model interpretability. Furthermore, the project emphasizes the practical application of the tools and concepts taught throughout the course, bridging the gap between theoretical knowledge and real-world data analysis.

Ultimately, this research not only contributes to a better understanding of air quality dynamics but also provides a foundation for further exploration of advanced machine learning techniques in environmental data analysis.

## Models used : 
GRU and LSTM

## MLflow Evaluation
In this section, we evaluate the MLflow models and deployments. MLflow is an open-source platform for end-to-end machine learning lifecycles. In this research, we have connected MLflow to our native code and deployed models on it. 


The model is trained in Jupyter Notebooks and then all model details including its parameters, metrics and status is uploaded into MLflow for evaluation. Using MLflow’s dynamic user interface, we can compare metrics of different runs and their parameters.



# Results
## Model Iteration 1:
Sequential Model: A sequential GRU model is built step by step.
•	First LSTM Layer: The first LSTM layer has 50 units (neurons) and uses the ReLU activation function. Return_sequences=True ensures that this layer outputs a sequence, which is required when stacking multiple LSTM layers.
•	Second LSTM Layer: Another LSTM layer with 50 units, but return_sequences=False (default), meaning it only outputs the last hidden state.
•	Dense Layer: A dense (fully connected) layer with 1 unit, which outputs the final prediction (PM2.5 value).
Compilation: The model is compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function, suitable for regression tasks.
Observations and results:

![image](https://github.com/user-attachments/assets/b6eeb3d3-0328-4bcb-a9f8-010d3e1fe99b)

Actual PM2.5 values: [4.49371360e-02 2.20850072e+02 3.85978488e+02 … 1.06653119e+02
 0.00000000e+00 3.16430564e-02]
Predicted PM2.5 values: [  2.91672705 215.2127437  519.72859228 …  99.00973004   1.15362336
   2.37400424]


With the first iteration of the modelling, we can see that the model performed relatively well with an epoch of 50. The predicted and actual values show a similar trend throughout the training phase except a few spikes in data where the predicted value is relatively very high as compared to the actual values. 
This is an anomaly in the result which should be fixed as we train the model in the upcoming iterations.

##  Model Iteration 2:
In this iteration, we slow down the learning rate of the optimizer so the model can learn better through the cycles. This is performed by the following line of code 
model.compile(optimizer=Adam(learning_rate=1e-4), loss=’mse’)

This slower learning rate will allow the model to learn the data slower and might avoid overshooting of the optimal solution. This process takes a bit long to complete with epoch = 50 , with a average cycle time of 200s. The net error seems to decrease from 0.3xx to 0.15x as we iterate through the cycles.
###  Observations and results:
4903/4903 ━━━━━━━━━━━━━━━━━━━━ 11s 2ms/step – loss: 0.0155
Test Loss: 0.015548395924270153
4903/4903 ━━━━━━━━━━━━━━━━━━━━ 12s 2ms/step

It can be observed that the Test Loss has considerably reduced as compared to the first iteration of the model training. Hence slowing the learning rate of the optimizer proves to be fruitful. 
Actual PM2.5 values: [4.49371360e-02 2.20850072e+02 3.85978488e+02 … 1.06653119e+02
 0.00000000e+00 3.16430564e-02]
Predicted PM2.5 values: [-3.78331088e-01  2.17488746e+02  2.44346687e+02 …  1.05927797e+02
  5.82028923e+00  1.18060000e-01]

The above statistic shows the sample values of predicted vs actual PM2.5 values.
 ![image](https://github.com/user-attachments/assets/6bc66084-966d-463f-a4b6-ebbace36277e)
Figure. Zoomed in Actual vs Predicted PM2.5 values for Iteration 2

Comparing the above images with the first iteration of the model training, we observe that the spikes of predicted PM2.5 values have been cut down. Though the issue of negative spiking persists with our model. 
The Fig.19 shows a zoomed trend of the Predicted PM2.5 vs Actual PM2.5. Upon further analysis, we observe that the model is learning and seems to be following the actual pattern.
## Model Iteration 3

For this iteration, we use a Bidirectional LSTM i.e. a LSTM that can process the data forward and backward. Hence the model can understand context from both directions. 
By substituting the forward and backward sequences for each hidden sequence and making sure that every hidden layer receives input from both the forward and backward layers at the level below, deep bidirectional RNNs may be put into practice. (Alex Graves, 2024)		

BiLSTM overcomes the limitations of the previous iteration by processing the input data of a neuron in both directions, forward and backward. This allows the model to access forward and backward contexts simultaneously, thereby leading to a more comprehensive understanding of the data.
Due to its complexity, BiLSTM requires more computing power but their improved ability to capture intricacies in data outweigh the expense.

### Observations and Results: 
4903/4903 ━━━━━━━━━━━━━━━━━━━━ 14s 3ms/step – loss: 0.0076
Test Loss: 0.007579026743769646
4903/4903 ━━━━━━━━━━━━━━━━━━━━ 16s 3ms/step

Actual PM2.5 values: [4.49371360e-02 2.20850072e+02 3.85978488e+02 … 1.06653119e+02
 0.00000000e+00 3.16430564e-02]
Predicted PM2.5 values: [-5.11730894e-01  1.78022658e+02  3.88194665e+02 …  1.02441757e+02
  3.04636440e+00  9.50391367e-02]

![image](https://github.com/user-attachments/assets/41854b5d-1693-4e0b-b28b-0501cdadc9ca)

Figure.Zoomed in Actual vs Predicted PM2.5 values for Iteration 3



##  Model Iteration 4
 In this iteration, we add a Dropout layer to our Bidirectional LSTM model and keep the other parameters like the previous iteration. Adding a dropout layer helps in generalizing random inputs inside the neural networks. During training, neural net activations are multiplied by random zero-one masks (random masks are not employed during test time; instead, they are approximated by a fixed scale).

###  Observations and Results: 
Actual PM2.5 values: [4.49371360e-02 2.20850072e+02 3.85978488e+02 … 1.06653119e+02
 0.00000000e+00 3.16430564e-02]
Predicted PM2.5 values: [  0.71900418 124.02480562 435.92137653 …  95.23619708   1.53055654
   0.92209861]

![image](https://github.com/user-attachments/assets/4fcd3145-858b-4a85-bf9d-61ab91cbc001)

Figure. Zoomed in Actual vs Predicted PM2.5 values for Iteration 4
##  Model Iteration 5
For this iteration, the model has the following features:
•	First LSTM Layer: The first LSTM layer has 100 units (neurons) and uses the ReLU activation function. Return_sequences=True ensures that this layer outputs a sequence, which is required when stacking multiple LSTM layers.
•	Second LSTM Layer: Another LSTM layer with 100 units, but return_sequences=False (default), meaning it only outputs the last hidden state.
•	Dense Layer: A dense (fully connected) layer with 1 unit, which outputs the final prediction (PM2.5 value).
### Observations and Results:
4903/4903 ━━━━━━━━━━━━━━━━━━━━ 19s 4ms/step – loss: 0.0010
Test Loss: 0.0010295321699231863
4903/4903 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step

Actual PM2.5 values: [4.49371360e-02 2.20850072e+02 3.85978488e+02 … 1.06653119e+02
 0.00000000e+00 3.16430564e-02]
Predicted PM2.5 values: [6.28881883e-01 1.88055144e+02 4.03734283e+02 … 1.16683403e+02
 4.59657530e+00 3.24108254e-01]

 
![image](https://github.com/user-attachments/assets/5a0450b1-af08-4e14-a432-c129e1aa3e6b)
Figure. Zoomed in Actual vs Predicted PM2.5 values for Iteration 5

