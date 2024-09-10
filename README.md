### **Facial Recognition Application using VGG-Face and LFW Dataset**
This project is a facial recognition application built using Tkinter for the user interface, and it leverages the VGG-Face model to compare a user-uploaded image with the faces from the Labeled Faces in the Wild (LFW) dataset. The application finds the most similar face from the dataset and returns the match.
![cap](https://github.com/user-attachments/assets/d018bc73-9f96-411a-90b7-459ff200009e)

# **Feature**:

Upload an image via a Tkinter-based interface

Uses the pre-trained VGG-Face model to generate facial embeddings

Compares the uploaded image to the LFW dataset using cosine similarity

Displays the person from the dataset who most closely resembles the uploaded image

Efficient nearest neighbor search with FAISS for large-scale facial comparisons
