from flask import Flask, render_template, request
import cv2
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import numpy as np
from tensorflow.keras.models import load_model
import time

app = Flask(__name__, template_folder="templates")
model = load_model("animal.h5")
print(model)

# SMTP Email Configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'Sridhar .R'
SMTP_PASSWORD = 'dlrkafsrvfbotamp'
SENDER_EMAIL = 'sridharspidey@gmail.com'
RECIPIENT_EMAIL = 'sridharspidey@gmail.com'

def send_email(subject, message, attachment=None):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject

    body = MIMEText(message)
    msg.attach(body)

    if attachment:
        img_data = open(attachment, 'rb').read()
        image = MIMEImage(img_data, name=os.path.basename(attachment))
        msg.attach(image)

    try:
        smtp_server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        smtp_server.starttls()
        smtp_server.login(SMTP_USERNAME, SMTP_PASSWORD)
        smtp_server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        smtp_server.quit()
        print("Email sent successfully")
    except smtplib.SMTPException as e:
        print("Failed to send email:", str(e))

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/home.html', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/intro.html', methods=['GET'])
def about():
    return render_template('intro.html')

@app.route('/upload.html', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/uploader.html', methods=['GET', 'POST'])
def predict():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        while True:
            (grabbed, frame) = cap.read()
            if not grabbed:
                break
            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (64, 64))
            x = np.expand_dims(frame, axis=0)
            result = np.argmax(model.predict(x), axis=1)
            index = ['Elephant', 'Lion', 'None', 'Tiger']
            result_str = str(index[result[0]])
            cv2.putText(output, "Activity: {}".format(result_str), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            cv2.imshow("Output", output)
            
            if result_str == index[0]:
                send_email("Animal Activity", "Detected animal: Elephant")
                time.sleep(3)
            elif result_str == index[1]:
                send_email("Animal Activity", "Detected animal: Lion")
                time.sleep(3)
            elif result_str == index[3]:
                send_email("Animal Activity", "Detected animal: Tiger")
                time.sleep(3)
                key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    time.sleep(2)  

    print("[INFO] Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()

    return render_template("upload.html")

