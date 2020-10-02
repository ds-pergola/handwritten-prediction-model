import flask
import predict
from PIL import Image
import io
import os
from google.cloud import storage

#GCP Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth/storage-admin.json"

app = flask.Flask(__name__)

#Model variable
sudoku = None

DEBUG = True

@app.route("/predict", methods=["POST"])
def api_predict():
    if DEBUG: print("Start test_parameters")

    # initialize the data dictionary that will be returned from the api
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            transaction_id = flask.request.form['transaction_id']

            # read the image from POST request
            image = flask.request.files["image"].read()

            #predicted_digit = sudoku.predict_local('3.png')

            # classify the input image
            predicted_digit = sudoku.predict_memory(io.BytesIO(image))

            data["predicted_digit"] = predicted_digit
            data["predicted_prob"] = None
            data["model_version"] = sudoku.version
            data["transaction_id"] = transaction_id

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    print(data)
    return flask.jsonify(data)

@app.route("/switch_model", methods=["POST"])
def switch_model():
    if DEBUG: print("Start test_parameters")
    
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.form.get("model_filename"):
            if flask.request.form.get("model_version"):
                
                model_filename = flask.request.form["model_filename"]
                model_version = flask.request.form["model_version"]

                if download_modal(model_filename):

                    if os.path.exists("model/" + model_filename):
                        os.environ['SUDOKU_MODEL_FILE_NAME'] = model_filename
                        os.environ['SUDOKU_MODEL_VERSION'] = model_version

                        global sudoku
                        sudoku = predict.SudokuPredict()
                        data["success"] = True

                    else:
                        data['message'] = "Model File couldn't downloaded to local({})".format(model_filename)
                else:
                        data['message'] = "Model File doesn't exist on model_storage({})".format(model_filename)

            else:
                data['message'] = "Model Version parameter not found(model_version)"
        else:
            data['message'] = "Model Filename parameter not found(model_filename)"

    return flask.jsonify(data)

@app.route("/test_parameters", methods=["POST"])
def test_parameters():
    if DEBUG: print("Start test_parameters")
    data = {"success": True}
    data["model_version"] = sudoku.version
    data["model_filename"] = sudoku.model_file_name
    data["tf_version"], data["tf_compiler_version"] = sudoku.get_tensorflow_parameters()

    return flask.jsonify(data)

def download_modal(model_filename):
    try:
        client = storage.Client() 
        bucket = client.get_bucket("dsp-sudoku")
        blob = bucket.blob('model_storage/' + model_filename)
        blob.download_to_filename('model/' + model_filename)
        print(f"Model downloaded from model_storage: {model_filename}")
        return True
    except:
        return False

def init():
    print(("Loading Keras model and Flask starting server..."
            "please wait until server has fully started"))
    print(f"ENV SUDOKU_MODEL_FILE_NAME: {os.environ['SUDOKU_MODEL_FILE_NAME']}")
    print(f"ENV SUDOKU_MODEL_VERSION: {os.environ['SUDOKU_MODEL_VERSION']}")

    global sudoku
    sudoku = predict.SudokuPredict()

#Used while testing locally with flask
if __name__ == "__main__":
    os.environ['SUDOKU_MODEL_FILE_NAME'] = 'sudoku_v1.h5'
    os.environ['SUDOKU_MODEL_VERSION'] = 'v1'

    init()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False, use_reloader=False)

#Used while running on Docker with gunicorn
if __name__ == "app" :
    init()
