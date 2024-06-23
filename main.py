import io
from PIL import Image
import cv2
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO  # Import your YOLO module

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_img():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(filepath)
            
            file_extension = f.filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                # Ensure img is not None before proceeding
                if img is not None:
                    cv2.imwrite(filepath, img)
                    # Now you can open the image using PIL
                    image = Image.open(filepath)
                    # Perform the detection
                    yolo = YOLO('best.pt')
                    detections = yolo.predict(image, save=True)
                    
                    # Construct JSON response for each detection result
                    json_responses = []
                    for detection in detections:
                        json_response = {
                            'filename': f.filename,
                            'orig_shape': detection.orig_shape,
                            # Include other relevant attributes here
                        }
                        json_responses.append(json_response)
                    
                    # Redirect to result page passing JSON responses
                    return redirect(url_for('show_results', predictions=json_responses))
                else:
                    # Redirect to error page if image reading failed
                    return redirect(url_for('show_error', error_message='Failed to read image'))
            else:
                # Redirect to error page if file extension is not jpg
                return redirect(url_for('show_error', error_message='File format not supported'))
                
    # Render the upload form for GET requests
    return render_template('index.html')

@app.route('/result', methods=['GET']) 
def show_results():
    # Get JSON responses from the URL parameter
    predictions = request.args.get('predictions')
    return render_template('result.html', predictions=predictions)

@app.route('/error', methods=['GET']) 
def show_error():
    # Get error message from the URL parameter
    error_message = request.args.get('error_message')
    return render_template('error.html', error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
