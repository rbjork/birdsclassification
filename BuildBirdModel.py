import os, PIL 
from PIL import Image
import pdb 
import random 
import numpy as np
import tensorflow as tf 
from tensorflow import keras


class Tree():
    def __init__(self,root):
        self.root = root
        self.leaves = []
        self.children = []
        self.mapping = {}
        self.size = 0
        with open('image_class_labels.txt','r') as fr:
            for line in fr:
                self.size += 1
                label, index = line.split(' ')
                index = index.strip()
                if index in self.mapping:
                    self.mapping[index].append(label)
                else:
                    self.mapping[index] = [label]

    def addNode(self,obj):
        self.children.append(obj)

    def getLeaves(self,node,depth):
        #print('depth',depth)
        children = node.children
        leaves = []
        if len(children) > 0: 
            for child in children:
                lvs = self.getLeaves(child,depth+1)
                leaves = leaves + lvs
            return leaves
        else:
            #print('leaf',node.data)
            #node.data = self.mapping[node.data]
            return [node] 

class Node():
    def __init__(self, data):
        self.data = data
        self.birdname = ''
        self.children = []
        self.images = []
        self.imagepaths =  []

    def addNode(self,obj):
        self.children.append(obj)
                # Sparrows, Finches, Chickadees, Thrushes
nodes_interest = ['212','231','69'] #['212','231','69','165']
family = {}  # sparrows, finches, chickadees (titmouse)

def getImage(image):
    im = Image.open(f'images/{image}')
    return im


import random 

def train(train_data,train_labels):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # Compile Model
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    # Fit Model
    print("fitting model")
    model.fit(train_images, train_labels, epochs=10)
    print("done fitting. saving")
    model.save('my_model')
    return model 

def training_data():
    train_images = []
    train_labels = []
    cnt = 0
    for d in os.listdir("image_reg"):
        for f in os.listdir(f"image_reg/{d}"):
            train_images.append(f"image_reg/{d}/{f}")
            train_labels.append(d)
            cnt += 1
    deck = list(range(0,cnt))
    random.shuffle(deck)
    # returns train images, train labels, test images, test labels
    return [train_images[i] for i in deck[0:-100]], [train_labels[i] for i in deck[0:-100]],[train_images[i] for i in deck[-100:]], [train_labels[i] for i in deck[-100:]]

def assembleData():
    global family 
    with open('bounding_boxes.txt','r') as fr:
        bbmap = {}
        for line in fr:
            label, x, y, width, height = line.split(' ')
            bbmap[label] = (int(x), int(y), int(width), int(height.strip()))

    with open('images.txt','r') as fr:
        filemap = {}
        for line in fr:
            label, path = line.split(' ')
            path = path.strip()
            filemap[label] = path 

    with open('classes.txt','r') as fr:
        classmap = {}
        for line in fr:
            vals = line.split(' ')
            birdclass = vals[0]
            name = ' '.join(vals[1:])
            birdclass = birdclass.strip()
            classmap[birdclass] = name 

    with open('hierarchy.txt','r') as fr:
        nodes = dict();
        parent_nodes = dict()
        root = True 
        for line in fr:
            #print(line);
            child, parent = line.split(' ')
            child = child.strip()
            parent = parent.strip()
            if root:
                tree = Tree(parent)
                root = False
                child_node = Node(child)
                parent_nodes[parent] = tree 
                tree.addNode(Node(child))
                print("Tree created")
                continue 
            else:
                if not parent in parent_nodes:
                    parent_node = Node(parent)
                    parent_nodes[parent] = parent_node 
                if not child in parent_nodes:
                    child_node = Node(child)
                    parent_nodes[child] = child_node
                pn = parent_nodes[parent]
                child_node = parent_nodes[child]
                pn.addNode(child_node)
                #print(parent.strip())
                if parent == '212':
                    print("adding child node ",child,'to',parent,'which has',len(pn.children),'children')
        #print(len(parent_nodes))
        #print(tree.mapping)
        print('tree.size',tree.size)
        
        leaves = []
        for ni in nodes_interest:
            pn = parent_nodes[ni]
            tvs = tree.getLeaves(pn,0)
            leaves = leaves + tvs
            family.update({t.data:ni for t in tvs})
        print(len(leaves))
        image_count = 0
        for leaf in leaves:
            bird = classmap[leaf.data]
            if leaf.data in tree.mapping:
                leaf.images = tree.mapping[leaf.data]
                print(bird, leaf.data, len(leaf.images))
                image_count += len(leaf.images)
                for img in leaf.images:
                    filepath = filemap[img]
                    leaf.imagepaths.append(filepath)

        print(image_count)
        targ_sz = 256  # 240 if EfficientNetB1 is used as starting model to be fine tuned
        #train_images = []
        #train_labels = []
        for i in range(0,len(leaves)):
            leaf = leaves[i]
            # j = int(random.uniform(0, len(leaf.images)))
            # num_images = min(5,len(leaf.images)) # **** throttle back ****
            for j in range(0,len(leaf.images)):
                id = leaf.data 
                label = leaf.images[j]
                leaf.birdname = classmap[id]
                im = getImage(leaf.imagepaths[j])
                #pdb.set_trace()
                x,y,w,h = bbmap[label]
                
                if int(w) < targ_sz or int(h) < targ_sz:
                    continue
                r = x + w 
                b = y + h 
                imc = im.crop((x,y,r,b))
                #pdb.set_trace()
                if w > h:
                    hsz = int(h * (targ_sz/w))
                    imc_sz = imc.resize((targ_sz,hsz), Image.Resampling.NEAREST)
                else:
                    wsz = int(w*(targ_sz/h))
                    imc_sz = imc.resize((wsz,targ_sz), Image.Resampling.NEAREST)

                # im_tf = tf.keras.utils.image.img_to_array(imc_sz)
                # OR
                im_na = np.array(imc_sz)
                if w > h:
                    padding = targ_sz - hsz
                    im_napad = np.pad(im_na, [(0, padding), (0, 0), (0, 0)], mode='constant')
                else:
                    padding = targ_sz - wsz
                    im_napad = np.pad(im_na, [(0, 0), (0, padding), (0, 0)], mode='constant')
            
                im_new = Image.fromarray(im_napad)
                fid = family[id]
                if not os.path.exists(os.path.join(f'image_reg256/{fid}')):
                    os.mkdir(f'image_reg256/{fid}')
                im_new.save(f'image_reg256/{fid}/{label}.jpg')

def testModel():
    model = tf.keras.models.load_model('birdmodel128')
    data_dir = "image_reg"
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=(128, 128),
        batch_size=10
    )

    test_loss, test_acc = model.evaluate(val_ds)
    print(test_acc)

def predict(imgpath):
    img = tf.keras.utils.load_img(
        imgpath, target_size=(128, 128)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    # imtf = tf.convert_to_tensor(
    #     ima, dtype=np.int8, dtype_hint=np.int8, name=None
    # )
    model = tf.keras.models.load_model('birdmodel128')
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    

if __name__ == "__main__":
    #train_images, train_labels, test_images, test_labels = training_data()
    #for d in os.listdir(f"image_reg")
    #assembleData()
    #exit()
    #predict("image_reg/997/001c81f1-d302-4029-8cb5-4488e966eff7.jpg")
    #exit() 

    data_dir = 'image_reg256'
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(256, 256),
        batch_size=10
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(256, 256),
        batch_size=10
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print(len(train_ds))
    #exit()
    #pdb.set_trace()
    #deck = list(range(0,100))
    #   random.shuffle(deck)
    #train_ds = train_ds# .takerandom(10)
    #val_ds = val_ds# .takerandom(10)
    num_classes = len(os.listdir('image_reg256')) #62

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5  # 3
    )

    model.save("birdmodel256")
    
        
