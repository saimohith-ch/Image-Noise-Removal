#importing all libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from PIL import Image
#loading a dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#adding noise to the image
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#train a model
autoencoder = Model(input_img, x)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()
autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
#input the image to evulate the results
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

input_image = load_and_preprocess_image('/content/testing.jpg')

# Add noise to the image
input_image_noisy = input_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_image.shape)
input_image_noisy = np.clip(input_image_noisy, 0., 1.)
denoised_image1 = autoencoder.predict(x_test_noisy)
denoised_image = autoencoder.predict(input_image_noisy)
#displaying the original image
n=10
plt.figure(figsize=(20,4))
for i in range(n):
  ax=plt.subplot(2,n,i+1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
#displaying the noise image
import tensorflow as tf
plt.figure(figsize=(20,4))
for i in range(n):
  ax=plt.subplot(1,n,i+1)
  plt.title("noise")
  plt.imshow(tf.squeeze(x_test_noisy[i]))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
#displaying the denoising image
n=10
plt.figure(figsize=(20,4))
for i in range(n):
  ax=plt.subplot(2,n,i+1)
  plt.imshow(denoised_image1[i])
  plt.title("denoised")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
#displaying the evulation results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(input_image[0].reshape(28, 28), cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Noisy')
plt.imshow(input_image_noisy[0].reshape(28, 28), cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Denoised')
plt.imshow(denoised_image[0].reshape(28, 28), cmap='gray')
plt.show()
# Calculate loss for noisy input images
noisy_input_loss = autoencoder.evaluate(x_test_noisy, x_test, verbose=0)

# Calculate loss for denoised images
denoised_loss = autoencoder.evaluate(denoised_image1, x_test, verbose=0)

# Calculate loss percentage
loss_reduction_percentage = ((noisy_input_loss - denoised_loss) / noisy_input_loss) * 100

print("Loss reduction percentage:", loss_reduction_percentage)
t=100
print("accuracy=",t+loss_reduction_percentage)



