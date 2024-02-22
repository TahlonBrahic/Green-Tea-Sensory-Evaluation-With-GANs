import onnxruntime as ort
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os

# Initialize models
Model_RF_session, Model_MLP_session, Model_RNN_session, Chemical_Scaler_session, Sensory_Scaler_session = None, None, None, None, None
test_X, unscaled_test_X  = None, None
test_y, unscaled_test_Y = None, None

def load_tea_catechin_models():
    global Model_RF_session, Model_MLP_session, Model_RNN_session
    global test_X, unscaled_test_X 

    # Needed for portability to elastic beanstalk
    base_dir = os.path.join(os.path.dirname(__file__), '../../../source/')

    if Model_RF_session is None:
        Model_RF_session = ort.InferenceSession(os.path.join(base_dir, 'Model_RF.onnx'))
    if Model_MLP_session is None:
        Model_MLP_session = ort.InferenceSession(os.path.join(base_dir, 'Model_MLP.onnx'))
    if Model_RNN_session is None:
        Model_RNN_session = ort.InferenceSession(os.path.join(base_dir, 'Model_RNN.onnx'))

    Chemical_Scaler_session = ort.InferenceSession(os.path.join(base_dir, "chemical_scaler.onnx"))
    Sensory_Scaler_session = ort.InferenceSession(os.path.join(base_dir, "sensory_scaler.onnx"))

    test_X = pd.read_csv(os.path.join(base_dir, "test_X.csv"))
    unscaled_test_X = pd.read_csv(os.path.join(base_dir, "unscaled_test_X.csv"))

    test_y = pd.read_csv(os.path.join(base_dir, "test_y.csv"))
    unscaled_test_y = pd.read_csv(os.path.join(base_dir, "unscaled_test_y.csv"))

def create_tea_catechins_plot(feature_1_name, feature_2_name, model_name, y_pred):
    # Extracting features for the plot using the provided feature names


    feature_1_values = test_X[feature_1_name]
    feature_2_values = test_X[feature_2_name]
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=feature_1_values,
        y=feature_2_values,
        z=y_pred,
        mode='markers',
        marker=dict(
            size=5,
            color=y_pred,  # color points by predicted values
            colorscale='Plasma',  # Color scale
            opacity=0.6
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
        template="plotly_dark"  # Use dark theme for the plot
    )

    return fig

def predict(model, features):
    global Model_RF_session, Model_MLP_session, Model_RNN_session, Chemical_Scaler_session, Sensory_Scaler_session
    
    # Scale features
    scaled_features = scale(features, Chemical_Scaler_session)
    
    # Select the appropriate model session
    model_session = {
        'Random Forest': Model_RF_session,
        'Multilayer Perceptron': Model_MLP_session,
        'Recurrent Neural Network': Model_RNN_session
    }.get(model, None)
    
    if model_session is None:
        return "Error: Model not found"
    
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    
    # For RNN, ensure the input is reshaped appropriately
    if model == 'Recurrent Neural Network':
        scaled_features = scaled_features.reshape(1, -1, len(features))
    
    # Perform prediction
    prediction = model_session.run([output_name], {input_name: scaled_features.astype(np.float32)})[0]
    
    # Inverse scale the prediction if necessary
    original_scale_prediction = inverse_scale(prediction, Sensory_Scaler_session)
    
    return original_scale_prediction 

def scale(features, scaler_session=Chemical_Scaler_session):
    input_name = scaler_session.get_inputs()[0].name
    output_name = scaler_session.get_outputs()[0].name
    scaled_features = scaler_session.run([output_name], {input_name: features.astype(np.float32)})[0]
    return scaled_features

def inverse_scale(predictions, scaler_session=Sensory_Scaler_session):
    input_name = scaler_session.get_inputs()[0].name
    output_name = scaler_session.get_outputs()[0].name
    original_scale_predictions = scaler_session.run([output_name], {input_name: predictions.astype(np.float32)})[0]
    return original_scale_predictions


# Notes:
# The model only knows scaled data.
# Scale features coming in.
# Inverse scale predictions coming out.