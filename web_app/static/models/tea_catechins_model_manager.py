import onnxruntime as ort
import plotly.graph_objs as go
import os

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
    data = [go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines', name='test')]
    layout = go.Layout(title='Interactive Plot', xaxis=dict(title='X-axis'), yaxis=dict(title='Y-axis'))
    fig = go.Figure(data=data, layout=layout)
    return fig
