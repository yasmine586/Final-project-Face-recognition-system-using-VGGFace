# **Facial Recognition Application using VGG-Face and LFW Dataset**
![cap](https://github.com/user-attachments/assets/d018bc73-9f96-411a-90b7-459ff200009e)

This project is a facial recognition application built using Tkinter for the user interface, and it leverages the VGG-Face model to compare a user-uploaded image with the faces from the Labeled Faces in the Wild (LFW) dataset. The application finds the most similar face from the dataset and returns the match.


#### **Feature**:

Upload an image via a Tkinter-based interface

Uses the pre-trained VGG-Face model to generate facial embeddings

Compares the uploaded image to the LFW dataset using cosine similarity

Displays the name of the person from the dataset who most closely resembles the uploaded image

Efficient nearest neighbor search with FAISS for large-scale facial comparisons

### **Installation**:

Clone the repository:
```
git clone https://github.com/yasmine586/Final-project-Face-recognition-system-using-VGGFace.git
cd Final-project-Face-recognition-system-using-VGGFace
```

Create a virtual environment and activate it:

```
virtualenv venv
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate

```

Install the required packages:
```
pip install -r requirements.txt

```
## **Usage**
(this step can be skipped as it create lfw_index_embedding.idx.idx file)
```
python create_embedding.py

```
Run the python script that open the tkinter window
```
python lfwgui.py

```
There's also the jupyter notebook for further explanation







