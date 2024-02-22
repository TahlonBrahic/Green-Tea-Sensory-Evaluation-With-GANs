import onnxruntime as ort
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os

# Initialize models
Model_RF_session, Model_MLP_session, Model_RNN_session = None, None, None
test_X = pd.read_csv('test_X.csv')

def load_tea_catechin_models():
    global Model_RF_session, Model_MLP_session, Model_RNN_session

    # Needed for portability to elastic beanstalk
    base_dir = os.path.join(os.path.dirname(__file__), '../../../source/')

    if Model_RF_session is None:
        Model_RF_session = ort.InferenceSession(os.path.join(base_dir, 'Model_RF.onnx'))
    if Model_MLP_session is None:
        Model_MLP_session = ort.InferenceSession(os.path.join(base_dir, 'Model_MLP.onnx'))
    if Model_RNN_session is None:
        Model_RNN_session = ort.InferenceSession(os.path.join(base_dir, 'Model_RNN.onnx'))

def create_tea_catechins_plot(test_X, y_pred, feature_1_name, feature_2_name, model_name):
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
    global Model_RF_session, Model_MLP_session, Model_RNN_session

    # Convert features to ONNX format
    features_formatted = np.array(features, dtype=np.float32).reshape(1, -1)  
    
    # Model Selection
    if model == 'Random Forest':
        model_session = Model_RF_session
    elif model == 'Multilayer Perceptron':
        model_session = Model_MLP_session
    elif model == 'Recurrent Neural Network':
        model_session = Model_RNN_session
        features_formatted = features_formatted.reshape(1, -1, len(features)) # Added because RNN's expect an input of (batch_size, sequence_length, num_features)
    else:
        return "Error" # Default error handling
    
    input_name = model_session.get_inputs()[0].name  

    # Perform prediction
    result = model_session.run(None, {input_name: features_formatted}) # None retrieves all
    
    return result[0]  