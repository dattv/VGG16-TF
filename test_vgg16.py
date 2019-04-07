import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

from vgg16 import *

TEST_DAT_DIR = "./test_data"

if not os.path.isdir(TEST_DAT_DIR):
    print("ERROR there are no {} folder".format(TEST_DAT_DIR))
else:
    files = os.listdir(TEST_DAT_DIR)
    path = os.getcwd()

    img = []
    for f in files:
        file = os.path.join(path, TEST_DAT_DIR[2:])
        file = os.path.join(file, f)

        image = skimage.io.imread(file)
        image = image / 255.0
        assert (0 <= image).all() and (image <= 1.0).all()
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        short_edge = min(image.shape[:2])
        yy = int((image.shape[0] - short_edge) / 2)
        xx = int((image.shape[1] - short_edge) / 2)
        crop_img = image[yy: yy + short_edge, xx: xx + short_edge]
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (224, 224))

        image = resized_img.reshape((1, 224, 224, 3))
        img.append(image)

    print(img[0].shape)
    print(img[1].shape)
    batch = np.concatenate((img[0], img[1]), 0)

    model = vgg16("./vgg16.npy")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        images = tf.placeholder("float", shape=[2, 224, 224, 3])

        with tf.name_scope("content_vgg"):
            # model.build_graph(images)
            model.build_graph(images)

        prob = sess.run(model.prob, feed_dict={images: batch})

        # prob = []
        #
        # for i in range(len(img)):
        #     print(i)
        #     prob.append(sess.run(model.prob, feed_dict=img[i]))

    SYNSET_DIR = './synset.txt'
    synset = [l.strip() for l in open(SYNSET_DIR).readlines()]

    title0 = np.argsort(prob[0])[::-1]
    top1_title0 = synset[title0[0]]

    title1 = np.argsort(prob[1])[::-1]
    top1_title1 = synset[title1[0]]

    img0 = np.reshape(img[0], [224, 224, 3])
    img1 = np.reshape(img[1], [224, 224, 3])
    plt.subplot(1, 2, 1).set_title(top1_title0)
    plt.imshow(img0)
    plt.subplot(1, 2, 2).set_title(top1_title1)
    plt.imshow(img1)

    plt.show()
