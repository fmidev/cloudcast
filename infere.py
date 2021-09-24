from tensorflow.keras.models import load_model
#import datasets, models, layers
from PIL import Image
from model import *
import glob

def read_img(filename):
    input_dir =f'/home/partio/tmp/cloudnwc-jpeg/'
    print(input_dir)

    files = glob.glob(f'{input_dir}/train/*.jpg')

    files_ds = tf.data.Dataset.from_tensor_slices(files)

    def process_img(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=(256, 256))
        return img

    files_ds = files_ds.map(lambda v: (process_img(v))).batch(1)

    return files_ds



#m = unet(input_size=(256,256,1))

#m.load_weights('checkpoints/cp.ckpt')

m = load_model('cloudcast.model')

predictions = m.predict(read_img('20210101T0030.jpg'), verbose=1)

pred=predictions[0].flatten().reshape(256,256)
pred=pred*256

#print(predictions[0].shape)
#print(predictions[0].flatten().shape)
print(np.min(pred),np.mean(pred),np.max(pred))
im = Image.fromarray(pred)
im.show()

#loss, acc = m.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


