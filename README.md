# Traffic Volume Prediction using LSTM (PyTorch)

This project builds a **deep learning time-series forecasting model using PyTorch** to predict traffic volume on a highway. The model uses historical traffic, weather, and time-related features to estimate the number of vehicles passing a location during the next time step.

Traffic prediction models can help address real-world challenges such as:

- traffic congestion
- urban road design
- transportation planning
- intelligent traffic systems
- commute optimization

---

# Project Overview

Traffic patterns are influenced by multiple factors including weather conditions, time of day, and holidays. Predicting these patterns accurately can improve transportation infrastructure and reduce congestion.

This project implements a **Long Short-Term Memory (LSTM) neural network** to learn temporal patterns in traffic data and forecast future traffic volume.

The model processes sequential data and predicts the **traffic volume for the next time step** based on previous observations.

---

# Dataset

The dataset originates from the **UCI Machine Learning Repository** and contains hourly traffic volume measurements from the **Interstate 94 highway in Minnesota, USA**.

Two datasets are used in this project:

```
train_scaled.csv
test_scaled.csv
```

These datasets have already been **scaled and prepared for modeling**.

---

# Features in the Dataset

The dataset includes weather conditions, time variables, and traffic measurements.

Examples of features include:

| Feature | Description |
|--------|-------------|
| temp | Average temperature |
| rain_1h | Rainfall during the hour |
| snow_1h | Snowfall during the hour |
| clouds_all | Cloud coverage percentage |
| hour_of_day | Hour of the day |
| day_of_week | Day of the week |
| day_of_month | Day of the month |
| month | Month of the year |
| weather features | Encoded weather descriptions |

The target variable is:

```
traffic_volume
```

which represents the **number of vehicles recorded per hour**.

---

# Time-Series Modeling

This project uses a **sequence-based learning approach**.

Each prediction is based on the previous **12 time steps**:

```
sequence length = 12
```

This allows the model to capture temporal dependencies and learn traffic patterns over time.

---

# Model Architecture

The model is a **Long Short-Term Memory (LSTM) neural network**.

Architecture:

```
Input sequence (12 timesteps × 66 features)
        ↓
2-layer LSTM (hidden size = 64)
        ↓
LeakyReLU activation
        ↓
Fully connected layer
        ↓
Predicted traffic volume
```

The model is implemented in:

```
model.py
```

and initialized as:

```
traffic_model
```

---

# Training Setup

The model is trained using:

**Loss Function**

```
Mean Squared Error (MSELoss)
```

**Optimizer**

```
Adam optimizer
learning rate = 0.0001
```

Training runs for:

```
2 epochs
```

The final training loss is stored as:

```
final_training_loss
```

---

# Evaluation

After training, the model is evaluated using **Mean Squared Error (MSE)** on the test dataset.

The evaluation result is stored as:

```
test_mse
```

Lower MSE values indicate better prediction performance.

---

# Files Included

| File | Description |
|-----|-------------|
| model.py | Defines the LSTM neural network used for traffic prediction |
| train.py | Trains the LSTM model using the training dataset |
| evaluate.py | Evaluates the trained model and calculates Mean Squared Error |
| train_scaled.csv | Preprocessed training dataset used for model training |
| test_scaled.csv | Preprocessed testing dataset used for model evaluation |
| README.md | Project documentation |

---

# Dependencies

Install the required Python libraries before running the project:

```
pip install torch pandas numpy
```

---

# Running the Project

### Train the model

```
python train.py
```

This script will:

- load the dataset
- train the LSTM model
- save the trained model

The trained model will be saved as:

```
traffic_model.pth
```

---

### Evaluate the model

```
python evaluate.py
```

This script will:

- load the trained model
- generate predictions on the test dataset
- compute Mean Squared Error

Example output:

```
Test MSE: <value>
```

---

# Outputs

Running the project generates:

```
traffic_model.pth
```

This file contains the **trained PyTorch model** and is generated automatically during training.

---

# Results

After training the model and evaluating it on the test dataset, the model produces a **Mean Squared Error (MSE)** score.

Example output:

```
Test MSE: <value>
```

Lower MSE values indicate better prediction performance.

This result demonstrates the model’s ability to capture **temporal traffic patterns using sequential deep learning**.

---

# Notes

Because the training configuration uses only **2 epochs**, the model may not achieve optimal performance. Increasing the number of training epochs or tuning hyperparameters can improve prediction accuracy.

This project demonstrates how **deep learning models can be applied to time-series forecasting problems such as traffic prediction**.
