import numpy as np
import cv2
import glob
from sys import argv
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle


class Obj:
    def __init__(self, name,result):
        self.name = name
        self.result=result

def error():
    print('\n# usage: python3 reco-images.py  ak [k1] [k2] [verbose]   (Apprentissage Kmeans)')
    print('#        python3 reco-images.py  rk [verbose]               (Reconnaissance Kmeans)')
    print('#        python3 reco-images.py  arl [k1] [k2] [verbose]    (Apprentissage Regression Logistique)')
    print('#        python3 reco-images.py  rrl [verbose]              (Reconnaissance Regression Logistique)\n')
    print('#        python3 reco-images.py  svm [verbose]              (SVM)\n')
    exit(1)
################################################
# usage: python3 reco-images.py  ak [k1] [k2] [verbose]   (Apprentissage Kmeans)
#        python3 reco-images.py  rk [verbose]             (Reconnaissance Kmeans)
#        python3 reco-images.py  arl [k1] [k2] [verbose]  (Apprentissage Regression Logistique)
#        python3 reco-images.py  rrl [verbose]            (Reconnaissance Regression Logistique)


################################################
#                                              #
#                                              #
#               Méthode K-MEANS                #
#                                              #
#                                              #
################################################

def apprentissage_kmeans(listImg,k1,k2,verbose):

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
            print("SIFT: ", keypoints)
        dimImg.append(len(descriptors))
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


def reconnaissance_kmeans(i,verbose):
    # lecture
    with open("kmean1", "rb") as input:
        kmeans1saved = pickle.load(input)

    k1 = kmeans1saved.n_clusters

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

    return prediction



################################################
#                                              #
#                                              #
#       Méthode Régression Logistique          #
#                                              #
#                                              #
################################################

def apprentissage_RegressionLogistique(k1,verbose):

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
    with open('bowsReg', 'wb') as output:
        pickle.dump(bows, output, pickle.HIGHEST_PROTOCOL)


def reconnaissance_RegressionLogistique(imageToTest,verbose):
    # lecture
    with open("kmean1", "rb") as input:
        kmeans1saved = pickle.load(input)

    k1 = kmeans1saved.n_clusters

    lesSifts = np.empty(shape=(0, 128), dtype=float)  # array of all SIFT
    dimImg = []
    # nb of SIFT per file

    for i in imageToTest:
        if verbose:
            print("###", i, "###")

        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        # compute sift
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if verbose:
            print("nb of keypoints: ", keypoints)

        dimImg.append(len(descriptors))
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
    return prediction


################################################
#                                              #
#                                              #
#                   SVM                        #
#                                              #
#                                              #
################################################

def svm():
    # pip3 install scikit-learn
    from sklearn import svm
    with open("bowsReg", "rb") as input:
        bows = pickle.load(input)
    X = bows
    y = [0,1]
    classif = svm.SVC(kernel='linear')
    classif.fit(X, y)
    labelsPredicted =classif.predict(bows)
    print('prediction class for [2,2]', labelsPredicted)
    print('support vectors: ', classif.support_vectors_)
    print("Confusion matrix :\n", confusion_matrix(y, labelsPredicted))
################################################
#                                              #
#                                              #
#                   MAIN                       #
#                                              #
#                                              #
################################################

#Nos Images références
listImg = glob.glob("train_motos/*.jpg")
nbMotos = len(listImg) # Nombre d'élements dans la classe moto
listImg += glob.glob("train_voitures/*.jpg")

labels = [0]*nbMotos # On remplie label avec autant de 0 que de motos
nbVoitures = len(listImg)-nbMotos # Nombre d'elements dans la classe voiture
labels += [1]*nbVoitures # On remplie label avec autant de 1 que de voitures

if len(argv)>4:
    if argv[4] == "True":
        log = True
    else:
        log = False
elif len(argv)>2:
    if argv[2] == "True":
        log = True
    else:
        log = False
elif len(argv)>1:
    log=False
else:
    error()
#Nos Images a testé
listImgTest = glob.glob("test/*.jpg")

if argv[1] == "ak":
    apprentissage_kmeans(listImg,int(argv[2]),int(argv[3]),log)
elif argv[1] == "rk":
    classes=[]
    objs=[]
    for img in listImgTest:
        prediction=reconnaissance_kmeans(img,log)
        if prediction not in classes:
            classes.append(prediction)
        objs.append(Obj(img,prediction))

    nbClasse=0
    for classe in classes:
        print("#####Groupe "+str(nbClasse)+"######")
        for obj in objs:
            if obj.result==classe:
                print(obj.name+" ")
        nbClasse+=1        
elif argv[1] == "arl":
    apprentissage_RegressionLogistique(int(argv[2]),log)
elif argv[1] == "rrl":
    prediction=reconnaissance_RegressionLogistique(listImgTest,log)
elif argv[1] == "svm":
    svm(log)
else:
    error();


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
