import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

# print(train_images[0])
print(train_label[0])

image_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
# print(image_labels.get(train_label[0]))

# plt.imshow(train_images[0], cmap='gray', )
# plt.show()

nn_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # tf.nn.softmax
])
nn_model.compile(tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')
nn_model.fit(train_images, train_label, epochs=5)

loss = nn_model.evaluate(test_images, test_label)
print(loss)

total = len(test_label)
# print(total)
wrong_predictions = 0

predicted_label = nn_model.predict(test_images)
# print(predicted_label[0])

for i in range(total-1): 
    predicted_label_n = list(predicted_label[i]).index(max(predicted_label[i]))

    # print("Predicted:", predicted_label_n)
    # print("Actual:", test_label[i])
    # print(image_labels.get(test_label[i]))

    # plt.imshow(test_images[i], cmap='gray', vmin=0, vmax=200)
    # plt.show()

    if predicted_label_n != test_label[i]:
      wrong_predictions +=1 

accuracy = (total - wrong_predictions) / total * 100
print(f'Accuracy: {accuracy:.0f}%')