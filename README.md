# 🎙️ Mmeslay

> An Automatic Speech Recognition System for the Kabyle language with a Flutter frontend.
>  - Live Demo Available at <https://huggingface.co/spaces/g1ya/Mmeslay>

---

## 📦

The backend Code has been moved to this repository :
<https://github.com/G1ya777/Mmeslay_backend-CLI>

## 📌 Overview

- This was created by training the **Squeezeformer-XS** model using the Common Voice dataset (Kabyle subset).
- The model was trained, validated, and tested on a custom split of the dataset.
- A language model was also trained on a text corpus composed of sentences collected from various sources, such as:
  - Tatoeba
  - https://github.com/MohammedBelkacem/Kabyletexts  
    using **KenLM**.
- The system was tested using various configurations of the CTC decoder:

<p align="center">
  <img src="https://github.com/G1ya777/Mmeslay/assets/116036106/c2723de3-6ee3-4ffb-a164-3fd489eef2e4" alt="table_git"/>
</p>

---

## 📚 Cite Us

If you use our project in your work, please cite:

```bibtex
@mastersthesis{Mmeslay,
  author  = {Aomer Gaya Ouldali},
  school  = {Université A. Mira de Béjaïa},
  title   = {Système de reconnaissance de la parole appliqué à la langue Tamazight},
  year    = {2023}
}
```
