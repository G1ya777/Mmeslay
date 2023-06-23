# Mmeslay
* An Automatic Speech Recognition System for the Kabyle language.
* This was created by training the Squeezeformer model using the Common Voice (the subset in kabyle language).
* The model was trained, validated and tested on a custom split of the dataset.
* A language model was also trained on a text corpus composed of sentences collected from various sources, such as Tatoeba and https://github.com/MohammedBelkacem/Kabyletexts.
* The system was tested using various configuration of the CTC decoder :
  
  ![table_git](https://github.com/G1ya777/Mmeslay/assets/116036106/c2723de3-6ee3-4ffb-a164-3fd489eef2e4)



# Prerequisites
* Run
`pip install Cython` then `pip install -r requirements.txt`
to install the dependencies.
* The app was tested on python 3.9.

# How to use
* To start the backend, cd into the backend folder and run `python src/flask_server.py`
* Run `flutter pub get` in the root of the frontend folder to install the required dependencies.
