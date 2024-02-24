import os  # Standard library imports
import numpy as np  # Third-party imports
import pandas as pd
import onnxruntime as ort
import plotly.graph_objs as go

# Initialization of global variables to None. These will be set by `load_tea_catechin_models`.
Model_RF_session, Model_MLP_session, Model_RNN_session = None, None, None
Chemical_Scaler_session, Sensory_Scaler_session = None, None
test_X, unscaled_test_X = None, None
test_y, unscaled_test_y = None, None
y_pred_rf, y_pred_mlp, y_pred_rnn = None, None, None

def load_tea_catechin_models():
    """
    Loads the ONNX models for Random Forest, MLP, RNN, and scalers for chemical and sensory data.
    Additionally, loads test datasets and their unscaled versions for evaluation purposes.
    Sets global variables for each model and dataset for use throughout the application.
    
    Raises:
        Exception: If there is an issue loading the models or datasets, an error message is printed.
    """
    global Model_RF_session, Model_MLP_session, Model_RNN_session
    global Chemical_Scaler_session, Sensory_Scaler_session
    global test_X, unscaled_test_X
    global test_y, unscaled_test_y
    global y_pred_rf, y_pred_mlp, y_pred_rnn

    base_dir = os.path.join(os.path.dirname(__file__), '../../../source/')

    try:
        # Load ONNX models for tea catechin prediction and associated scalers
        Model_RF_session = ort.InferenceSession(os.path.join(base_dir, 'Model_RF.onnx'))
        Model_MLP_session = ort.InferenceSession(os.path.join(base_dir, 'Model_MLP.onnx'))
        Model_RNN_session = ort.InferenceSession(os.path.join(base_dir, 'Model_RNN.onnx'))
        Chemical_Scaler_session = ort.InferenceSession(os.path.join(base_dir, "chemical_scaler.onnx"))
        Sensory_Scaler_session = ort.InferenceSession(os.path.join(base_dir, "sensory_scaler.onnx"))

        # Load test datasets and their unscaled versions
        test_X = pd.read_csv(os.path.join(base_dir, "test_X.csv"))
        unscaled_test_X = pd.read_csv(os.path.join(base_dir, "unscaled_test_X.csv"))
        test_y = pd.read_csv(os.path.join(base_dir, "test_y.csv"))
        unscaled_test_y = pd.read_csv(os.path.join(base_dir, "unscaled_test_y.csv"))

        # Load model predictions 
        y_pred_rf = np.load(os.path.join(base_dir, 'y_pred_rf.npy'))
        y_pred_mlp = np.load(os.path.join(base_dir, 'y_pred_mlp.npy'))
        y_pred_rnn = np.load(os.path.join(base_dir, 'y_pred_rnn.npy'))

    except Exception as e:
        print(f"Error loading models or data: {e}")

# Prediction
def predict(model, features):    

    features_array = np.array(features, dtype=np.float32).reshape(1, -1)
    model_session = {'Random Forest': Model_RF_session, 
                     'Multilayer Perceptron': Model_MLP_session, 
                     'Recurrent Neural Network': Model_RNN_session}.get(model, None)
    
    if model_session is None:
        print("Error: Model not found")
        return None
    
    # Scale the input
    try:
        scaled_features = scale(features_array, Chemical_Scaler_session)
        if scaled_features is None:
            print("Error: Could not scale features")
            return None
        
    except Exception as e:
        print(f"Error scaling features for {model}: {e}")
        return None

    try:
        input_name = model_session.get_inputs()[0].name
        output_name = model_session.get_outputs()[0].name
        
        # Correctly using scaled_features for prediction
        if model == 'Recurrent Neural Network':
            # Make sure to use scaled_features for RNN input preparation
            rnn_input = prepare_rnn_input(scaled_features)
            prediction = model_session.run([output_name], {input_name: rnn_input})[0]
        else:
            prediction = model_session.run([output_name], {input_name: scaled_features})[0]

        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
    
        # Inverse scale the prediction
        prediction = np.array(prediction, dtype=np.float32)
        prediction = inverse_scale(prediction, Sensory_Scaler_session)
        
        print(f"Prediction: {prediction}")  
        return prediction
    except Exception as e:
        print(f"Error making prediction for {model}: {e}")
        return None

# Interactive Plot
import numpy as np
import plotly.graph_objs as go

def create_tea_catechins_plot(feature_1_name='Catechin', feature_2_name='Caffeine', model_name='Random Forest'):
    global unscaled_test_X, y_pred_rf, y_pred_mlp, y_pred_rnn

    load_tea_catechin_models()

    # Choose prediction data based on the selected model
    if model_name == 'Random Forest':
        y_pred = y_pred_rf
    elif model_name == 'Multilayer Perceptron':
        y_pred = y_pred_mlp
    elif model_name == 'Recurrent Neural Network':
        y_pred = y_pred_rnn
    else:
        raise ValueError("Invalid model name")
  
    # Extracting features for the plot using the provided feature names
    feature_1_values = unscaled_test_X[feature_1_name].values
    feature_2_values = unscaled_test_X[feature_2_name].values
    
    # Flatten y_pred to ensure it's one-dimensional
    y_pred = y_pred.flatten()

    # Debugging: Check shapes of the arrays after flattening y_pred
    print(f"Shapes - feature_1: {feature_1_values.shape}, feature_2: {feature_2_values.shape}, y_pred: {y_pred.shape}")
    if not (len(feature_1_values) == len(feature_2_values) == len(y_pred)):
        raise ValueError("Data length mismatch after flattening!")

    # Cleaning NaN or infinite values in y_pred
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    # Debugging: Print sample values to ensure data is as expected
    print(f"Sample y_pred values: {y_pred[:5]}")

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=feature_1_values,
        y=feature_2_values,
        z=y_pred,
        mode='markers',
        marker=dict(
            size=5,
            color=y_pred,  
            colorscale='Plasma',
        )
    )])

    # Update plot layout to use feature names in axis titles
    fig.update_layout(
        title=f'3D Scatter Plot of Sensory Evaluation for {model_name}',
        scene=dict(
            xaxis_title=feature_1_name,
            yaxis_title=feature_2_name,
            zaxis_title='Sensory Evaluation',
            xaxis=dict(backgroundcolor="rgb(200, 200, 200)", color='white'),
            yaxis=dict(backgroundcolor="rgb(200, 200, 200)", color='white'),
            zaxis=dict(backgroundcolor="rgb(200, 200, 200)", color='white')
        ),
        template="plotly_dark"
    )

    return fig


# Utility Functions
def scale(features, scaler_session):
    try:
        input_name = scaler_session.get_inputs()[0].name
        output_name = scaler_session.get_outputs()[0].name
        features_array = np.array(features).astype(np.float32).reshape(-1, 9)
        print(f"Min feature value: {np.min(features_array)}, Max feature value: {np.max(features_array)}")
        scaled_features = scaler_session.run([output_name], {input_name: features_array})[0]
        print(f"Min scaled feature value: {np.min(scaled_features)}, Max scaled feature value: {np.max(scaled_features)}")
        return scaled_features
    except Exception as e:
        print(f"Error scaling features: {e}")
        return None

def inverse_scale(predictions, scaler_session):
    try:
        input_name = scaler_session.get_inputs()[0].name
        output_name = scaler_session.get_outputs()[0].name
        original_scale_predictions = scaler_session.run([output_name], {input_name: predictions.astype(np.float32)})[0]
        return original_scale_predictions
    except Exception as e:
        print(f"Error inverse scaling predictions: {e}")
        return None

def prepare_rnn_input(features, target_sequence_length=2000, num_features=9):
    if features.ndim == 1:
        features = features.reshape(-1, num_features)
    padded_input = np.zeros((target_sequence_length, num_features), dtype=np.float32)
    sequence_length = min(features.shape[0], target_sequence_length)
    padded_input[:sequence_length, :] = features[:sequence_length, :]
    return padded_input.reshape(1, target_sequence_length, num_features)

# Example inputs
# .25, .22, .06, .05, .17, .17, .05, .04, .11
# Example output
# .86
