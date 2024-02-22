import onnxruntime as ort
import plotly.graph_objs as go
import os
import numpy as np

# Initialize models
Model_RF_session, Model_MLP_session, Model_RNN_session = None, None, None

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

def create_tea_catechins_plot():
    # This is filler for an actual interactive plot
    data = [go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines', name='test')]
    layout = go.Layout(title='Interactive Plot', xaxis=dict(title='X-axis'), yaxis=dict(title='Y-axis'))
    fig = go.Figure(data=data, layout=layout)
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
    else:
        return "Error" # Default error handling
    
    input_name = model_session.get_inputs()[0].name  

    # Perform prediction
    result = model_session.run(None, {input_name: features_formatted}) # None retrieves all
    
    return result[0]  