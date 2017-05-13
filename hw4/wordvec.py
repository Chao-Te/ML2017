import word2vec
import numpy as np
import nltk

##word2vec.word2phrase('./data/all.txt' , './res/all-phrases',verbose=True)
##
##word2vec.word2vec('./res/all-phrases', './res/all.bin',size=100, verbose=True)
##
##word2vec.word2clusters('./data/all.txt','./res/all_cluster.txt', 100, verbose=True)

import matplotlib
matplotlib.use('Agg')

doTrain=False
doTest=True
corpus='./data/all.txt'
mdoel='./all.bin'
img_name='hp.png'
num_fword=800

if doTrain:
    # DEFINE your parameters for training
    WORDVEC_DIM = 200                 #Set size of word vectors; default is 100
    MIN_COUNT = 5                   #This will discard words that appear less than <int> times; default is 5
    WINDOW = 10                     #Set max skip length between words; default is 5
    NEGATIVE_SAMPLES = 5       #Number of negative examples; default is 0, common values are 5 - 10
    ITERATIONS = 50                #number of iterations
    LEARNING_RATE = 0.025          #Set the starting learning rate; default is 0.025
    
    # train model
    word2vec.word2vec(
        train=corpus,
        output=mdoel,
        size=WORDVEC_DIM,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        iter_=ITERATIONS,
        alpha=LEARNING_RATE,
        verbose=True)
if doTest:
    # load model for plotting
    model = word2vec.load(mdoel)

    vocabs = []                 
    vecs = []                   
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:num_fword]
    vocabs = vocabs[:num_fword]

    '''
    Dimensionality Reduction
    '''
    # from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)


    '''
    Plotting
    '''
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
    
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    plt.savefig(img_name, dpi=600)
    plt.show()
