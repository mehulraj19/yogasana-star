# yogasana-star
This is project I have worked with my teammates in one of the courses in my college. Our simple objective was to motivate our peers to handle stress and other issues during Covid times
using yoga at home.

# Project Development
Technologies I have used:
<ul>
  <li>FLask framework</li>
  <li>SQL Database</li>
  <li>MLP Classifier</li>
  <li>Random Forest Classifier</li>
  <li>SVC from svm</li>
  <li>StandardScalar</li>
  <li>Pipeline</li>
  <li>Open-cv</li>
  <li>front-end: HTML, Bootstrap, Custom CSS</li>
  <li>Jinja for python in front-end</li>
</ul>

# Working of the Project
<ul>
  <li>open-cv for the data pre-processing and for showing results in case of webcam</li>
  <li>Random forest Classifier model for image classification</li>
  <li>SVC model from svm, used pipeline to first scale the parameters before giving them input for model training for image classification</li>
  <li>Out of these models, I have preferred to use SVM since it has given me better accuarcy and then have used this for yoga pose detection via image and for scoring</li>
  <li>MLP Classsifier model for yoga pose detection via webcam.</li>
  <li>Simple HR-Net model for estimating the pose, used COCO model and it's weigts for pose detection</li>
  <li>Flask framework for backend-web application and to be able to import models and implement in web-app</li>
  <li>SQL Database in back-end</li>
  <li>HTML, bootstrap, Custom CSS and Jinja for front-end</li>
</ul>

# Run this project
For running this project, clone this project and then <a href="https://github.com/bbdavidbb/YogAi">click here</a> to access all the files for the yoga pose detection via webcam. Then you may need to
install some of the libraries.
<br/>
You can use the following command in Anaconda prompt or termial of your choice:-

```
pip install <LibraryName>
```

After you have installed all the libraries, you can simply run the app using this command in your terminal:

```
python run.py
```

you can copy the following URL and paste in your browser to see the webpage:

```
http://127.0.0.1:5000/
```

# Future Works
I am working on the webcam page and making sure to have a better UI in that as well.

## Credits
bbdavidbb
