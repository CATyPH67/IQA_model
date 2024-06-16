import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk
from keras import layers
from keras import Model
from fftlayer import *
from keras import models
from keras import utils
import numpy
from keras import applications

class ImageQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Quality Assessment")

        self.model = self.load_model()

        self.canvas = tk.Canvas(root, width=256, height=256)
        self.canvas.pack()

        self.load_weights_button = tk.Button(root, text="Load Model Weights", command=self.load_weights)
        self.load_weights_button.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.predict_button = tk.Button(root, text="Get Quality Score", command=self.predict_quality)
        self.predict_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        self.image = None
        self.img_array = None

    def load_model(self):
        input_shape = (256, 256, 3)

        # base_model_space = models.load_model("base_model_space.h5")
        # base_model_freq = models.load_model("base_model_freq.h5")

        base_model_space = applications.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
            include_preprocessing=True,
        )

        base_model_freq = applications.Xception(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg"
        )

        # Разаморозка всех слоев предобученной модели
        for layer in base_model_space.layers:
            layer.trainable = True
        for layer in base_model_freq.layers:
            layer.trainable = True

        # Создание двухголовой гидры
        inputs = keras.Input(shape=input_shape)
        # Пространственная ветка
        x = base_model_space(inputs, training=True)
        # Частотная ветка
        fft_input = fftGradLayer(shape=input_shape)
        y = base_model_freq(fft_input(inputs), training=True)
        # Объединение карт признаков
        z = keras.layers.Concatenate()([x, y])
        outputs = layers.Dense(1)(z)

        base_model_space.trainable = True
        base_model_freq.trainable = True
        # Создание новой модели
        model = Model(inputs, outputs)

        model.load_weights("cp_model_efficientnetv2-s.weights.h5")
        return model

    def load_weights(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.model.load_weights(file_path)
            messagebox.showinfo("Weights Loaded", f"Model weights loaded from {file_path}")

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            input_shape = self.model.input.shape

            # lscale = layers.Rescaling(scale=1 / 127.5, offset=-1)
            lcrop = layers.CenterCrop(height=input_shape[2], width=input_shape[1])

            img = utils.load_img(file_path)

            self.image = ImageTk.PhotoImage(img)
            self.canvas.create_image(128, 128, image=self.image)
            self.result_label.config(text="")

            self.img_array = utils.img_to_array(img)
            self.img_array = lcrop([self.img_array])

            # img = Image.open(file_path).convert('RGB')
            # self.image = ImageTk.PhotoImage(img)
            # self.canvas.create_image(128, 128, image=self.image)
            # self.result_label.config(text="")
            # width, height = img.size
            # left = (width - target_size[0]) / 2
            # top = (height - target_size[1]) / 2
            # right = (width + target_size[0]) / 2
            # bottom = (height + target_size[1]) / 2
            # img = img.crop((left, top, right, bottom))
            # self.img_array = img_to_array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
            # self.img_array = np.expand_dims(self.img_array, axis=0)

    def predict_quality(self):
        if self.img_array is not None:
            koniq_range = numpy.array([1., 2., 3., 4., 5.], dtype='float32')
            lnorm = layers.Normalization(axis=None, invert=True)
            lnorm.adapt(koniq_range)

            prediction = self.model.predict(self.img_array)
            # print(prediction[0][0])
            prediction = lnorm(prediction)

            self.result_label.config(text=f"Predicted Quality Score: {prediction[0][0]:.2f}")
            # print(prediction[0][0])

            # prediction = self.model.predict(self.img_array)[0][0]
            # self.result_label.config(text=f"Predicted Quality Score: {prediction:.2f}")
        else:
            messagebox.showwarning("No Image", "Please upload an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageQualityApp(root)
    root.mainloop()
