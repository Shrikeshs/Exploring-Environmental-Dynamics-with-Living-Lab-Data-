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




