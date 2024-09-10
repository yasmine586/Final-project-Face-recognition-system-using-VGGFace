import tkinter as tk
from tkinter import filedialog
from sklearn.datasets import fetch_lfw_people
from deepface import DeepFace
import faiss
import cv2 as cv
import numpy as np
lfw = fetch_lfw_people(min_faces_per_person=5,resize=0.4,color=True)#more face per person means a better embedding
#color = True means that it will be 3 dimensional (RGB OR BGR)
#fetching the important information
images=lfw.images
target_names=lfw.target_names
targets=lfw.target 

index=faiss.read_index("lfw_index_embedding.idx") #refer to the notebook on how i got this embedding
def upload_image():
    global image_path #global variable so that even after leaving the function the image_path value will stay 
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]) #create an open dialog and return the selected filename(s) that correspond to existing file(s).
    try:#because for the functin DeepFace.represent the argument enforce_face detection is by default True and launch an error when the face is not detected
        most_similar_idx = find_most_similar_image(image_path, index)
        most_similar_person_name = target_names[targets[most_similar_idx]]
        label_text.config(text=f"Most similair person : {most_similar_person_name}")
    except:
        label_text.config(text=f"Face could not be detected ")

    
        
           
def find_most_similar_image(uploaded_image_path, faiss_index):
    
    uploaded_image = cv.imread(uploaded_image_path)
    if uploaded_image is None:
        raise ValueError(f"Could not read image from path: {uploaded_image_path}")
    
    # Convert the image to RGB (DeepFace expects RGB format)
    uploaded_image_rgb = cv.cvtColor(uploaded_image, cv.COLOR_BGR2RGB)
    
    
    uploaded_embedding = DeepFace.represent(uploaded_image_rgb)##do the embedding of the uploaded image by default the model will be "VGGFace"
    uploaded_embedding = np.array(uploaded_embedding[0]["embedding"], dtype='float32').reshape(1, -1)#faiss expect a numpy array
    
    # Search for the closest match in FAISS
    distances, indices = faiss_index.search(uploaded_embedding, k=1)
    #k represent the number of nearest neighbors retrived during a search
    #indices and distances will be a matrix but since k=1 there will be only one value (one nearest neighbour)
    #indices represent the index of the most closest vector to the uploaded embedding
    #distances is the distance between the embedding vector and the closest vector
    return indices[0][0]        
       

# Windows creation
WIDTH,HEIGHT=600,300 #tuple having the window dimension
window = tk.Tk()
window.title("lfw datasets look alike")
window.geometry('%sx%s'%(WIDTH,HEIGHT))


image_path = "" # variable that save the location of the uploaded image

# Bouton to uplaod the image
upload_button = tk.Button(window, text="upload", command=upload_image)
upload_button.pack(pady=20)  #pady is for the positioning of the button it will be in the center +20 for the y axis so a little bit up there


label_text = tk.Label(window, text="Please upload a picture (high quality so the face can be detected)", wraplength=380)
label_text.pack(pady=10)#same as the previous one but it will be +10 for the y axis so bellow the previous button

# main loop for our window
window.mainloop()

