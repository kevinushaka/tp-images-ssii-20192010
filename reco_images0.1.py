import glob
import pickle
from sys import argv

from pip._vendor.webencodings import labels
import matplotlib.pyplot as plt
#import shutil
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
import cv2
from skimage import io

################################################
def BOWs(k,dim,kmeans):
    #BOW initialization
    bows = np.empty(shape=(0,k),dtype=int)
    #writing the BOWs for second k-means
    i = 0
    for nb in dim: # for each sound (file)
        tmpBow = np.array([0]*k)
        j = 0
        while j < nb: # for each MFCC of this sound (file)
            tmpBow[kmeans.labels_[i]] += 1
            j+=1
            i+=1
        tmpBow = tmpBow / nb
        copyBow = tmpBow.copy()
        bows = np.append(bows, [copyBow], 0)
    return bows;

def write(kmeans,name):
    #écriture
    with open(name,'wb') as output:
        pickle.dump(kmeans,output,pickle.HIGHEST_PROTOCOL)

################################################
# usage: python3 recosons.py k1 k2 verbose
# ATTENTION: les noms de fichiers ne doivent comporter ni - ni espace

#sur ligne de commande: les 2 parametres de k means puis un param de verbose

k1 = int(argv[1])
k2 = int(argv[2])

if argv[3] == "True":
    verbose = True;
else:
    verbose = False;


listImages=glob.glob("motos/*.jpg")
tmpa = len(listImages) #on mémorise le nb d'éléments de la première classe
listImages += glob.glob("voitures/*.jpg")
#liste des labels:
groundTruth = [0]*tmpa
tmpb = len(listImages)-tmpa #nb. éléments de la snde classe
groundTruth += [1]*tmpb


lesSift = np.empty(shape=(0, 128), dtype=float) # array of all descriptors from all images
dimImages = [] # nb of descriptors per file

for s in listImages:
    if verbose:
        print("###",s,"###")
    image = io.imread(s)
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray,None)
    if verbose:
        print("SIFT: ", keypoints)
    dimImages.append(len(descriptors))
    lesSift = np.append(lesSift,descriptors,axis=0)


# everything ready for the 1st k-means
kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesSift)
if verbose:
    print("result of kmeans 1", kmeans1.labels_)
    
#on calcule les bows
bows=BOWs(k1,dimImages,kmeans1)

if verbose:
    print("nb of SWIFT vectors per file : ", dimImages)
    print("BOWs : ",bows )

#ready for second k-means
kmeans2 = KMeans(n_clusters=k2, random_state=0).fit(bows)
if verbose:
    print("result of kmeans 2", kmeans2.labels_)

write(kmeans1,"kmean1")
write(kmeans2,"kmean2")

#cŕeation d'un objet de regression logistique
logisticRegr = LogisticRegression()
#apprentissage
logisticRegr.fit(bows, groundTruth)
#calcul des labels pŕeditslabels
Predicted = logisticRegr.predict(bows)
#calcul et affichage du score
score = logisticRegr.score(bows, groundTruth)
print("train score = ", score)
#sauvegarde de l'objet
with open('sauvegarde.logr', 'wb') as output:
    pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)
#chargement de l'objet
with open('sauvegarde.logr',  'rb') as input:
    logisticRegr = pickle.load(input)

################################################
imageToTest = glob.glob(argv[4])
#lecture
with open("kmean1","rb") as input :
    kmeans1saved = pickle.load(input)

k1 = kmeans1saved.n_clusters

#On recupère le mfcc du son
lesSift = np.empty(shape=(0, 128), dtype=float) # array of all descriptors from all sounds
dimImages = [] # nb of descriptors per file

for s in imageToTest:
    if verbose:
        print("###",s,"###")
    image = io.imread(s)
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray,None)
    if verbose:
        print("SIFT: ", descriptors)
    dimImages.append(len(descriptors))
    lesSift = np.append(lesSift,descriptors,axis=0)


#On crée le premier kmean avec les mfcc
mfccpredict = kmeans1saved.predict(lesSift)

#on calcule les bows
bows = np.empty(shape=(0,k1),dtype=int)

#writing the BOWs for second k-means
i = 0
for nb in dimImages: # for each sound (file)
    tmpBow = [0]*k1
    j = 0
    while j < nb: # for each MFCC of this sound (file)
        tmpBow[mfccpredict[i]] += 1
        j+=1
        i+=1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)

if verbose:
    print("nb of SIFT vectors per file : ", dimImages)
    print("BOWs :\n", bows)

#On fait la prédiction avec logisticRegr
#lecture
with open("sauvegarde.logr","rb") as input :
    logisticRegr = pickle.load(input)

prediction = logisticRegr.predict(bows)
if verbose:
    print("result of logisticRegr", prediction)
