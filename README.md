# About This Repo
This is my first project using Yolo & Flask

I implement simple GUI web application that can recomend outfit based on user's clothes.
(Using weather is not implemented yet)

I crawling data from "musinsa.com" to train my tiny cody_recommend_model.

This tiny model based on ResNet50 embedding top & bottom iamge to 2048-dim and then concat to 4096-dim.
Finally Classifier predict either "Good Codi" or "Bad Codi" for all cloth pair in user's closet.

This idea was begining from Machine Learning's ensemble method. So, I want to mixing two model's prediction in one application.


# How to run

1. Clone this repo

```$ git clone https://github.com/basamg912/Clothing-Detection-and-Outfit-Recommendation-System.git
```

2. Activate conda environment 
```
$ cd Project
$ conda env create -f environment.yml
$ conda activate cody
```

3. Run this application
```
$ python app.py
```

4. Open your browser and go to 
http://localhost:5000