
from flask.globals import session
from yoga import app
from flask import render_template, redirect, url_for, flash, request
from yoga.models import User
from yoga.forms import RegisterForm, LoginForm
from yoga import db
from yoga import model, loaded_model
from flask_login import login_user, logout_user, login_required, current_user


import numpy as np
import cv2
import os
import sys
import argparse
import ast
import cv2
import torch
import glob
import time
import pickle
from sklearn.neural_network import MLPClassifier
from vidgear.gears import CamGear
sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict

import cv2
# model = pickle.load(open('yoga_model.pkl', 'rb'))

# Variables
IMG_SIZE = 200
COUNT = 0
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,100)
fontScale              = 1
fontColor              = (0,0,0)
lineType               = 2


# Routes
@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/leaderboard')
def leaderboard_page():
    leaders = User.query.order_by(User.score)
    return render_template('leaderboard.html', leaders=leaders)

@app.route('/choice')
@login_required
def choice_page():
    return render_template('choice.html')


@app.route('/testViaImage')
def practice_page():
    return render_template('practice.html')


## prediction via image
@app.route('/predict', methods=['POST'])
def resultForm():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    image = cv2.imread('static/{}.jpg'.format(COUNT))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, -1)
    pred = model.predict(img)
    data = pred[0]
    res = model.predict_proba(img)
    val = max(res[0])
    user = User.query.filter_by(username=current_user.username).first()
    if val > 0.6:
        user.score += 0.01
    else:
        user.score += 0.05
    db.session.commit()
    print(current_user.score)
    return render_template('result.html', data=data, val=val)

@app.route('/register', methods = ['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email=form.email.data,
                              password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f'Account created Successfully!! You are now logged in as {user_to_create.username}', category="success")
        return redirect(url_for('choice_page'))
    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'There was an error creating a user: {err_msg}', category='danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(attempted_password=form.password.data):
            login_user(attempted_user)
            flash(f'Success!! You are logged in as: {attempted_user.username}', category='success')
            return redirect(url_for('choice_page'))
        else:
            flash('Username or Password do not match, Please Try Again', category='danger')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout_page():
    logout_user()
    flash('You have been Logged Out!!', category='info')
    return redirect(url_for('home_page'))

@app.route('/webcam')
def open_app(camera_id = 0, filename = None, hrnet_c = 48, hrnet_j = 17, hrnet_weights = "./weights/pose_hrnet_w48_384x288.pth", hrnet_joints_set = "coco", image_resolution = '(384, 288)', single_person = True,max_batch_size = 16, disable_vidgear = False, device = None):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    if filename is not None:
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        resolution=image_resolution,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        device=device
    )
    
    no_to_label = {0:"tree", 1:"warrior1", 2:"warrior2", 3:"childs",4:"downwarddog",5:"plank",6:"mountain",7:"trianglepose"}
    image_to_blob = {}
    for id,path in no_to_label.items():
        images = [cv2.imread(file) for file in glob.glob('sampleposes\\'+path+'.jpg')]
        image_to_blob[id] = images
    while True:
        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
        else:
            frame = video.read()
            if frame is None:
                break
        pts = model.predict(frame)
        resolution = frame.shape
        x_len = resolution[0]
        y_len = resolution[1]
        vector = []
        if len(pts) == 0:
            continue
        keypoints = pts[0]

        for pt in keypoints:
            pt = list(pt)
            temp = []
            temp.append((pt[0]/x_len))
            temp.append((pt[1]/y_len))
            vector.extend(temp)

        vector = list(vector)
        predicted_pose = loaded_model.predict([vector]) 
        text = no_to_label[predicted_pose[0]] + " pose"
        cv2.putText(image_to_blob[predicted_pose[0]][0], text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType) 
        cv2.imshow("Suggestion",image_to_blob[predicted_pose[0]][0])
        k= cv2.waitKey(1)
        for i, pt in enumerate(pts):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)

        if has_display:
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                disable_vidgear=True
                if disable_vidgear:
                    video.stop()
                    return redirect(url_for('choice_page'))
                else:
                    video.stop()
                break
                
        else:
            cv2.imwrite('frame.png', frame)