from imageio import imread, imwrite
from util import *
import FlowNet_S as flownet_s

if __name__ == '__main__':
    image1 = imread(
        './images/image1.png')
    image1 = apply_transform(image1)
    image2 = imread(
         './images/image2.png')
    image2 = apply_transform(image2)
    image = np.concatenate([image1, image2], axis=-1) #for blue output background
    # image = np.concatenate([image2, image1], axis=-1) #for yellow output background
    image = np.reshape(image, (1, 384,512, 6))
    image = image.astype('float32')

    model = flownet_s.FlowNet_S()
    model.build(input_shape=(None, 384,512, 6))
    model.load_weights('full path to weights')
    model.compile()
    model.summary()
    output = model(image)
    div_flow = 20
    rgb_flow = flow2rgb(div_flow * output, max_flow=10)
    to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
    imwrite('./images/result' + '.png', to_save)
