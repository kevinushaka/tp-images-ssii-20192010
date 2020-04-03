import numpy as np
import cv2
import glob
from sys import argv
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle


################################################
#                                              #
#                                              #
#               Méthode K-MEANS                #
#                                              #
#                                              #
################################################

def apprentissage_kmeans():
    k1 = int(argv[2])
    k2 = int(argv[3])

    if argv[4] == "True":
        verbose = True
    else:
        verbose = False

    lesSifts = np.empty(shape=(0, 128), dtype=float)  # array of all SIFT
    dimImg = []  # nb of SIFT per file

    for i in listImg:
        if verbose:
            print("###", i, "###")

        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        name = i.replace('./samples\\', '')
        cv2.imwrite('./sift/SIFT_' + name, img)

        # compute sift
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if verbose:
            # print(descriptors[0])
            print("SIFT: ", descriptors.shape)
        dimImg.append(descriptors.shape[0])
        lesSifts = np.append(lesSifts, descriptors, axis=0)

    # BOW initialization
    bows = np.empty(shape=(0, k1), dtype=int)

    # everything ready for the 1st k-means
    kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesSifts)
    if verbose:
        print("result of kmeans 1", kmeans1.labels_)

    # writing the BOWs for second k-means
    i = 0
    for nb in dimImg:  # for each pic (file)
        tmpBow = [0] * k1
        j = 0
        while j < nb:  # for each SIFT
            tmpBow[kmeans1.labels_[i]] += 1
            j += 1
            i += 1
        copyBow = tmpBow.copy()
        bows = np.append(bows, [copyBow], 0)
    if verbose:
        print("nb of SIFT vectors per file : ", dimImg)
        print("BOWs :\n", bows)

    # ready for second k-means
    kmeans2 = KMeans(n_clusters=k2, random_state=0).fit(bows)



    if verbose:
        print("result of kmeans 2", kmeans2.labels_)

    # écriture
    with open("kmean1", 'wb') as output:
        pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)
    with open("kmean2", 'wb') as output:
        pickle.dump(kmeans2, output, pickle.HIGHEST_PROTOCOL)


def reconnaissance_kmeans(i):
    # lecture
    with open("kmean1", "rb") as input:
        kmeans1saved = pickle.load(input)

    k1 = kmeans1saved.n_clusters

    if argv[4] == "True":
        verbose = True
    else:
        verbose = False

    lesSifts = np.empty(shape=(0, 128), dtype=float)  # array of all SIFT from all pictures
    dimImg = []

    if verbose:
        print("###", i, "###")

    img = cv2.imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    # compute sift
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if verbose:
        print("Descriptors: ", descriptors.shape)

    dimImg.append(descriptors.shape[0])
    lesSifts = np.append(lesSifts, descriptors, axis=0)

    # On crée le premier kmean avec les SIFT
    siftpredict = kmeans1saved.predict(lesSifts)

    # on calcule les bows
    bows = np.empty(shape=(0, k1), dtype=int)

    # writing the BOWs for second k-means
    i = 0
    for nb in dimImg:  # for each pic (file)
        tmpBow = [0] * k1
        j = 0
        while j < nb:  # for each SIFT
            tmpBow[siftpredict[i]] += 1
            j += 1
            i += 1
        copyBow = tmpBow.copy()
        bows = np.append(bows, [copyBow], 0)
    if verbose:
        print("nb of SIFT vectors per file : ", dimImg)
        print("BOWs :\n", bows)

    # On fait la prédiction avec kmeans2saved

    # lecture
    with open("kmean2", "rb") as input:
        kmeans2saved = pickle.load(input)
    print(kmeans2saved.labels_)

    prediction = kmeans2saved.predict(bows)
    if verbose:
        print("result of kmeans 2", prediction)



################################################
#                                              #
#                                              #
#       Méthode Régression Logistique          #
#                                              #
#                                              #
################################################

def apprentissage_RegressionLogistique():
    k1 = int(argv[2])
    k2 = int(argv[3])

    if argv[4] == "True":
        verbose = True
    else:
        verbose = False

    lesSifts = np.empty(shape=(0, 128), dtype=float)  # array of all SIFT from all sounds
    dimImg = []  # nb of SIFT per file

    first = True
    for i in listImg:
        if verbose:
            print("###", i, "###")

        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        # compute sift
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if first:
            print(descriptors[0])
            first = False
        if verbose:
            print("SIFT: ", descriptors.shape)
        dimImg.append(descriptors.shape[0])
        lesSifts = np.append(lesSifts, descriptors, axis=0)

    # BOW initialization
    bows = np.empty(shape=(0, k1), dtype=int)

    # everything ready for the 1st k-means
    kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesSifts)
    if verbose:
        print("result of kmeans 1", kmeans1.labels_)

    # writing the BOWs for second k-means
    i = 0
    for nb in dimImg:  # for each pic (file)
        tmpBow = [0] * k1
        j = 0
        while j < nb:  # for each SIFT
            tmpBow[kmeans1.labels_[i]] += 1
            j += 1
            i += 1
        copyBow = tmpBow.copy()
        bows = np.append(bows, [copyBow], 0)
    if verbose:
        print("nb of SIFT vectors per file : ", dimImg)
        print("BOWs :\n", bows)

    logisticRegr = LogisticRegression(max_iter=300)
    # apprentissage
    logisticRegr.fit(bows, labels)
    # calcul des labels
    labelsPredicted = logisticRegr.predict(bows)
    # calcul score
    score = logisticRegr.score(bows, labels)

    print("train score = ", score)
    print("Confusion matrix :\n", confusion_matrix(labels, labelsPredicted))

    # sauvegarde de l'objet
    with open("kmean1", 'wb') as output:
        pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)
    with open('sauvegarde.logr', 'wb') as output:
        pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)


def reconnaissance_RegressionLogistique():
    # lecture
    with open("kmean1", "rb") as input:
        kmeans1saved = pickle.load(input)

    k1 = kmeans1saved.n_clusters

    if argv[4] == "True":
        verbose = True
    else:
        verbose = False

    lesSifts = np.empty(shape=(0, 128), dtype=float)  # array of all SIFT
    dimImg = []
    # nb of SIFT per file

    for i in listImgTest:
        if verbose:
            print("###", i, "###")

        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        # compute sift
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if verbose:
            print("Descriptors: ", descriptors.shape)

        dimImg.append(descriptors.shape[0])
        lesSifts = np.append(lesSifts, descriptors, axis=0)

    # On crée le premier kmean avec les SIFT
    siftpredict = kmeans1saved.predict(lesSifts)

    # calcule les bows
    bows = np.empty(shape=(0, k1), dtype=int)

    # writing the BOWs for second k-means
    i = 0
    for nb in dimImg:  # for each pic (file)
        tmpBow = [0] * k1
        j = 0
        while j < nb:  # for each SIFT
            tmpBow[siftpredict[i]] += 1
            j += 1
            i += 1
        copyBow = tmpBow.copy()
        bows = np.append(bows, [copyBow], 0)
    if verbose:
        print("nb of SIFT vectors per file : ", dimImg)
        print("BOWs :\n", bows)

    # On fait la prédiction avec logisticRegr
    # lecture
    with open("sauvegarde.logr", "rb") as input:
        logisticRegr = pickle.load(input)

    prediction = logisticRegr.predict(bows)
    if verbose:
        print("result of logisticRegr", prediction)


################################################
#                                              #
#                                              #
#                   MAIN                       #
#                                              #
#                                              #
################################################

#Nos Images références
listImg = glob.glob("motos/*.jpg")
nbMotos = len(listImg) # Nombre d'élements dans la classe moto
labels = [0]*nbMotos # On remplie label avec autant de 0 que de motos

listImg += glob.glob("voitures/*.jpg")
nbVoitures = len(listImg)-nbMotos # Nombre d'elements dans la classe voiture
labels += [1]*nbVoitures # On remplie label avec autant de 1 que de voitures

#Nos Images a testé
listImgTest = glob.glob("Motos et Voiture pour reconnaissance/*.jpg")

if argv[1] == "ak":
    apprentissage_kmeans()
elif argv[1] == "rk":
    reconnaissance_kmeans(argv[5])
elif argv[1] == "arl":
    apprentissage_RegressionLogistique()
elif argv[1] == "rrl":
    reconnaissance_RegressionLogistique()
else:
    print("argument Inconnu !")

'''

labels = [0,0,0,0,1,1,1,1,1,1]
#cr´eation d'un objet de regression logistique
logisticRegr = LogisticRegression()
#apprentissage
logisticRegr.fit(bows, labels)
#calcul des labels pr´edits
labelsPredicted = logisticRegr.predict(bows)
#calcul et affichage du score
score = logisticRegr.score(bows, labels)
print("train score = ", score)

print("Confusion matrix :\n", confusion_matrix(labels, labelsPredicted))
#sauvegarde de l'objet
'''
'''
with open('sauvegarde.logr', 'wb') as output:
    pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)
#chargement de l'objet
with open('sauvegarde.logr', 'rb') as input:
    logisticRegr = pickle.load(input)'''
