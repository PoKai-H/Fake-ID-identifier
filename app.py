from flask import Flask,render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
 

ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        f = request.files['file_upload']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "please check yout file type (only except JPG, PNG, jpg, png files)"})

        base_path = os.path.dirname(__file__)
        

        upload_path = os.path.join(base_path, 'images/upload_image', secure_filename(f.filename))
        f.save(upload_path)
        
        uploaded_img = Image.open(upload_path).resize((250,160))
        
        uploaded_img.save(os.path.join(base_path,'images/upload_image','tampered.jpg'))
        
        original_img = Image.open(os.path.join(base_path,'images/origin_image','origin1.jpg')).resize((250,160))
        original_img.save(os.path.join(base_path,'images/origin_image','origin1.jpg'))

        

        origin_img = cv2.imread(os.path.join(base_path,'images/origin_image','origin1.jpg'))
        upload_img = cv2.imread(os.path.join(base_path,'images/upload_image','tampered.jpg'))
        
        
        original_grey = cv2.cvtColor(origin_img,cv2.COLOR_BGR2GRAY)
        uploaded_grey = cv2.cvtColor(upload_img,cv2.COLOR_BGR2GRAY)

        # Compute the structural similarity Index (SSIM) between the two images, ensuring that the difference image is returned
        (score, diff)= structural_similarity(original_grey, uploaded_grey, full=True)
        diff = (diff * 255).astype("uint8")
        
        # Calculating threshold and contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        

        #loop over the contours
        for c in cnts:
        # applying contours on image 
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(origin_img, (x,y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(upload_img, (x,y), (x+y, y+h), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(base_path,'static/generated','image_original1.jpg'),origin_img)
        cv2.imwrite(os.path.join(base_path,'static/generated','image_uploaded1.jpg'),upload_img)
        cv2.imwrite(os.path.join(base_path,'static/generated','diff1.jpg'),diff)
        cv2.imwrite(os.path.join(base_path,'static/generated','thresh1.jpg'),thresh)

        return render_template('index_ok.html',pred=str(round(score*100,2))+ '%' + 'correct')
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8987, debug=False)