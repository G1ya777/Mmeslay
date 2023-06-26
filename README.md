# Mmeslay
* An Automatic Speech Recognition System for the Kabyle language with a flutter Frontend.
* This was created by training the Squeezeformer-XS model using the Common Voice (the subset in kabyle language).
* The model was trained, validated and tested on a custom split of the dataset.
* A language model was also trained on a text corpus composed of sentences collected from various sources, such as Tatoeba and https://github.com/MohammedBelkacem/Kabyletexts, using KenLM.
* The system was tested using various configuration of the CTC decoder :
  
  ![table_git](https://github.com/G1ya777/Mmeslay/assets/116036106/c2723de3-6ee3-4ffb-a164-3fd489eef2e4)



# Prerequisites
* Run
`pip install Cython` then `pip install -r requirements.txt`
to install the dependencies. Using Conda or something similar is recommeneded.
* Run `flutter pub get` in the root of the frontend folder to install the required dependencies.
* The app was tested on Python 3.9, Flutter 3.10 and Dart 3.0.

# How to use
* To start the backend, cd into the backend folder and run `python src/flask_server.py`
* Run `flutter pub get` in the root of the frontend folder to install the required dependencies.
* Build the front end from source or install the appended release .apk.
* The front end was only tested on Android, although, it should be able to run on other platforms supported by flutter, but it may need some tweaks
* The default ip adress for both the frontend and the backend is `192.168.12.1`, you can change it in the source code to suit your needs.

# Cite us
* If you use our project in your work, please cite us :
<pre>
<code>
@mastersthesis{Mmeslay,<br>
  author  = {Aomer Gaya Ouldali},<br>
  school  = {Université A. Mira de Béjaïa},<br>
  title   = {Système de reconnaissance de la parole appliqué à la langue Tamazight},<br>
  year    = {2023}<br>
}
</code>
</pre>
