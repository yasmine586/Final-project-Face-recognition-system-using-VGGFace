from sklearn.datasets import fetch_lfw_people
from deepface import DeepFace
import numpy as np
import faiss
lfw = fetch_lfw_people(min_faces_per_person=5,resize=0.4,color=True)#more face per person means a better embedding
#color = True means that it will be 3 dimensional (RGB OR BGR)
#fetching the important information
images=lfw.images
target_names=lfw.target_names
targets=lfw.target
#it's better to do all of this in a function
def generate_lfw_embeddings(lfw_images):
    embeddings = []
    for image in lfw_images:#going through each image
        embedding = DeepFace.represent(image, enforce_detection=False) #all the image in the lfw database have face so there's no need to check if there's one with enforce_detection ,we can use any model(FaceNet,OpenFace) by default they use VGG-Face
        embeddings.append(embedding[0]["embedding"])
    return np.array(embeddings,dtype='float32')#we need the embedding in form of numpy array not list because later on when we save the index in faiss (index.add(embeddings))they need to be numpyarray

lfw_embeddings = generate_lfw_embeddings(images)
# function to store the embedding in faiss
def store_embeddings_faiss(embeddings):
    
    dimension = embeddings.shape[1]  # Use the actual embedding dimensionality
    
    # Create faiss index with the nedded dimension
    index = faiss.IndexFlatL2(dimension) #leave the vector as it is (flat) and it's used to calculate the distance between the vector and l2 stand for the metrics used to calculate the distance (euclidian norm)
    

    
    # Add embeddings to the faiss index
    index.add(embeddings)
    
    return index
index = store_embeddings_faiss(lfw_embeddings)
faiss.write_index(index, "lfw_indxex_embedding.idx")
