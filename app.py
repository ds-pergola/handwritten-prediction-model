import flask
import predict
from PIL import Image
import io
import os

app = flask.Flask(__name__)
sudoku = None

@app.route("/predict", methods=["POST"])
def api_predict():
    # initialize the data dictionary that will be returned from the
    # view
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
            data["predicted_accuracy"] = None
            data["model_version"] = sudoku.version
            data["transaction_id"] = transaction_id

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route("/reload_model", methods=["POST"])
def reload_model():
    
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.form.get("model_filename"):
            if flask.request.form.get("model_version"):
                
                model_filename = flask.request.form["model_filename"]
                model_version = flask.request.form["model_version"]

                if os.path.exists("model/" + model_filename):
                        os.environ['SUDOKU_MODEL_FILE_NAME'] = model_filename
                        os.environ['SUDOKU_MODEL_VERSION'] = model_version

                        global sudoku
                        sudoku = predict.SudokuPredict()
                        data["success"] = True

                else:
                    data['message'] = "Model File not found({})".format(model_filename)

            else:
                data['message'] = "Model Version parameter not found(model_version)"
        else:
            data['message'] = "Model Filename parameter not found(model_filename)"

    return flask.jsonify(data)


@app.route("/test_parameters", methods=["POST"])
def test_parameters():
    data = {"success": True}
    data["model_version"] = sudoku.version
    data["model_filename"] = sudoku.model_file_name
    data["tf_version"], data["tf_compiler_version"] = sudoku.get_tensorflow_parameters()

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("Loading Keras model and Flask starting server..."
            "please wait until server has fully started"))
    
    #To be deleted. Env variables should flow from Dockerfile
    os.environ['SUDOKU_MODEL_FILE_NAME'] = 'sudoku_v1.h5'
    os.environ['SUDOKU_MODEL_VERSION'] = 'v1'

    sudoku = predict.SudokuPredict()

    app.run(debug=True, host='0.0.0.0', port=5001, threaded=False, use_reloader=False)