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
def create_tea_catechins_plot(feature_1_name='Catechin', feature_2_name='Caffeine', model_name='Random Forest'):
    global test_X
    
    # Generate predictions for all test_X data based on the selected model
    y_pred = generate_plot_predictions(model_name)
    
    # Extracting features for the plot using the provided feature names
    feature_1_values = unscaled_test_X[feature_1_name]  # Make sure to use unscaled_test_X for actual feature values
    feature_2_values = unscaled_test_X[feature_2_name]
    
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

def get_model_predictions(scaler_session, model_session, test_data):
    features = test_data.to_numpy(dtype=np.float32)
    
    # Scale features
    scaled_features = scale(features, scaler_session)
    if scaled_features is None:
        return None

    try:
        # Check if the model session corresponds to the RNN model
        if model_session == Model_RNN_session:
            # Correctly prepare the RNN input
            rnn_input = prepare_rnn_input(scaled_features, target_sequence_length=2000, num_features=scaled_features.shape[-1])
            input_name = model_session.get_inputs()[0].name
            output_name = model_session.get_outputs()[0].name
            # Use rnn_input instead of scaled_features for RNN predictions
            predictions = model_session.run([output_name], {input_name: rnn_input})[0]
        else:
            # For non-RNN models, proceed as before
            input_name = model_session.get_inputs()[0].name
            output_name = model_session.get_outputs()[0].name
            predictions = model_session.run([output_name], {input_name: scaled_features})[0]
        
        # Inverse scale predictions
        original_scale_predictions = inverse_scale(predictions, Sensory_Scaler_session)
        return original_scale_predictions.flatten() if original_scale_predictions is not None else None
    except Exception as e:
        print(f"Error getting model predictions: {e}")
        return None


def generate_plot_predictions(model_name):
    global Chemical_Scaler_session, Model_RF_session, Model_MLP_session, Model_RNN_session, test_X
    
    # Load models if not already done
    load_tea_catechin_models()
    
    model_session = {
        'Random Forest': Model_RF_session,
        'Multilayer Perceptron': Model_MLP_session,
        'Recurrent Neural Network': Model_RNN_session
    }.get(model_name, None)
    
    if model_session is None:
        raise ValueError("Model not found")
    
    # Generate predictions for all test_X data
    y_pred = get_model_predictions(Chemical_Scaler_session, model_session, test_X)
    
    return y_pred   

# Utility Functions
def scale(features, scaler_session):
    try:
        input_name = scaler_session.get_inputs()[0].name
        output_name = scaler_session.get_outputs()[0].name
        features_array = np.array(features).astype(np.float32).reshape(-1, 9)
        scaled_features = scaler_session.run([output_name], {input_name: features_array})[0]
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
