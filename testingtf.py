from tflite_runtime.interpreter import Interpreter

# Load the Edge TPU model
interpreter = Interpreter(model_path="model_edgetpu.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()

# Print the input dtype
print("Input data type:", input_details[0]['dtype'])
