import deeplake
from PIL import Image


data = deeplake.load("hub://activeloop/ffhq")[:5000]
images = data.tensors['images_1024/image'].numpy()
print(images.shape)

for i in range(5000):
    image = images[i]
    image = Image.fromarray(image)
    image.save(f'./ffhq/img_{i}.png', "png")


print("########### Image Download Ended ###########")
