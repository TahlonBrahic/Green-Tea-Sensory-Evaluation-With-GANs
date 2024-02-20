import onnxruntime as ort

# Initialize models
Model_RF_session, Model_MLP_session, Model_RNN_session = None, None, None

def load_tea_catechin_models():
    global Model_RF_session, Model_MLP_session, Model_RNN_session
    if Model_RF_session is None:
        Model_RF_session = ort.InferenceSession('source/Model_RF.onnx')
    if Model_MLP_session is None:
        Model_MLP_session = ort.InferenceSession('source/Model_MLP.onnx')
    if Model_RNN_session is None:
        Model_RNN_session = ort.InferenceSession('source/Model_RNN.onnx')

