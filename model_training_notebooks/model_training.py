def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 4]  # Index 4 corresponds to 'PM2.5'
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Defining sequence length
sequence_length = 10

# Creating sequences from the scaled data
X, y = create_sequences(scaled_data, sequence_length)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------
# Model Training: GRU Model
# ---------------------------------------

# Model hyperparameters
num_epochs = 50
learning_rate = 0.001
batch_size = 32
validation_split = 0.2
input_shape = (sequence_length, X_train.shape[2])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Building the GRU model
gru_model = Sequential()
gru_model.add(GRU(50, activation='relu', return_sequences=True, input_shape=input_shape))
gru_model.add(GRU(50, activation='relu'))
gru_model.add(Dense(1))

# Compiling the model
gru_model.compile(optimizer='adam', loss='mse')

# Training the GRU model
gru_history = gru_model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)

# ---------------------------------------
# Model Training: Bidirectional LSTM Model
# ---------------------------------------

from tensorflow.keras.layers import Bidirectional, LSTM

# Building the Bidirectional LSTM model
bilstm_model = Sequential()
bilstm_model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=input_shape))
bilstm_model.add(Bidirectional(LSTM(50, activation='relu')))
bilstm_model.add(Dense(1))

# Compiling the Bidirectional LSTM model with different hyperparameters
bilstm_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

# Training the Bidirectional LSTM model
bilstm_history = bilstm_model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)

# ---------------------------------------
# Model Evaluation and Prediction
# ---------------------------------------

# Evaluating the Bidirectional LSTM model on the test set
test_loss = bilstm_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Predicting using the trained Bidirectional LSTM model
y_pred = bilstm_model.predict(X_test)

# Rescaling the predictions and actual values to the original scale
y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], df_cleaned.shape[1]-1)), y_test.reshape(-1, 1)), axis=1))[:, -1]
y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], df_cleaned.shape[1]-1)), y_pred), axis=1))[:, -1]

# Displaying the actual vs predicted PM2.5 values
print("Actual PM2.5 values:", y_test_rescaled)
print("Predicted PM2.5 values:", y_pred_rescaled)

# Plotting actual vs predicted PM2.5 values
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, color='blue', label='Actual PM2.5')
plt.plot(y_pred_rescaled, color='red', label='Predicted PM2.5')
plt.title('Actual vs Predicted PM2.5')
plt.xlabel('Samples')
plt.ylabel('PM2.5')
plt.legend()
plt.show()

# Calculating Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Zoomed-in plot for a specific sample range
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled[100:150], color='blue', label='Actual PM2.5')
plt.plot(y_pred_rescaled[100:150], color='red', label='Predicted PM2.5')
plt.title('Zoomed in Actual vs Predicted PM2.5')
plt.xlabel('Samples')
plt.ylabel('PM2.5')
plt.legend()
plt.show()

# ---------------------------------------
# Logging the Model and Parameters with MLflow
# ---------------------------------------

from datetime import datetime

# Defining the experiment and run names
experiment_name = "LSTM_MODEL_TRAINING"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=run_name) as mlflow_run:
    
    # Setting experiment tags
    mlflow.set_experiment_tag("base_model", "LSTM")
    mlflow.set_tag("optimizer", "keras.optimizers.Adam")
    mlflow.set_tag("loss", "mse")

    # Logging the model with MLflow
    mlflow.keras.log_model(bilstm_model, "model")

    # Logging parameters and metrics
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("input_shape", input_shape)

    mlflow.log_metric("train_loss", bilstm_history.history["loss"][-1])
    mlflow.log_metric("val_loss", bilstm_history.history["val_loss"][-1])

    # Logging run ID
    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)
