import os
import time
import numpy as np
import json

def loadAndCreateGloveModel(dim):
    glove_path = "./../../Pretrained-Embedding/GloVe/"
    path = os.path.join(glove_path, "glove.6B."+str(dim)+"d.txt")
    start = time.time()
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    print("Loaded GloVe at %s in %fs" % (os.path.basename(path), time.time()-start))
    
    lines = data.split('\n')
    print("There are %d lines." % len(lines))
    
    start = time.time()
    model = {}
    for line in lines:
        tokens = line.split()
        try:
            model[tokens[0]] = np.array([np.float32(x) for x in tokens[1:]])
        except IndexError as e:
            print("Index Error for: ", tokens)
            print(e)
    print("Model GloVe "+os.path.splitext(os.path.basename(path))[0]+" created in %fs" % (time.time()-start))
    save_path = os.path.join(glove_path, "glove.6B."+str(dim)+"d.json")
#    with open(save_path, 'w', encoding='utf-8') as f:
#        json.dump(model, f)
    return model
