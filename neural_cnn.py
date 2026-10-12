import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import time


# ============================================================
# LIVE NEURAL NETWORK VISUALIZER
# Callback ini HANYA MEMBACA model (forward pass training=False),
# tidak mengubah bobot, loss, optimizer, ataupun data training.
# Jadi proses & hasil training tetap sama seperti kode asli.
# ============================================================
class NeuralNetVisualizer(tf.keras.callbacks.Callback):
    LAYER_NAMES = ["Input\n28x28", "Conv1\n32 filter", "Conv2\n64 filter",
                   "Conv3\n128 filter", "Dense\n128", "Output\n0-9"]
    NODES = [12, 8, 8, 8, 10, 10]  # jumlah neuron yang DITAMPILKAN per layer

    def __init__(self, x_samples, y_samples, update_every=50):
        super().__init__()
        self.x_samples = x_samples.astype("float32")
        self.y_samples = y_samples
        self.update_every = update_every  # update tampilan tiap N batch
        self.k = 0
        self.epoch = 0
        self.enabled = True
        self.hist = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}
        self._build_figure()

    # ---------- membuat jendela & geometri jaringan ----------
    def _build_figure(self):
        plt.ion()
        bg = "#0b0f1a"
        self.fig = plt.figure(figsize=(15, 7.5), facecolor=bg)
        try:
            self.fig.canvas.manager.set_window_title("Neural Network Training Monitor")
        except Exception:
            pass

        gs = self.fig.add_gridspec(3, 4, hspace=0.5, wspace=0.3)
        self.ax_net = self.fig.add_subplot(gs[:, :3])
        self.ax_img = self.fig.add_subplot(gs[0, 3])
        self.ax_acc = self.fig.add_subplot(gs[1, 3])
        self.ax_loss = self.fig.add_subplot(gs[2, 3])
        for ax in (self.ax_net, self.ax_img, self.ax_acc, self.ax_loss):
            ax.set_facecolor(bg)

        # posisi neuron
        n_layers = len(self.NODES)
        xs = np.linspace(0.08, 0.92, n_layers)
        self.pos = []
        for li, n in enumerate(self.NODES):
            ys = np.linspace(0.12, 0.80, n)
            self.pos.append([(xs[li], y) for y in ys])

        # garis koneksi antar layer (urutan: i-major lalu j, cocok dengan np.outer)
        segs = []
        for li in range(n_layers - 1):
            for p in self.pos[li]:
                for q in self.pos[li + 1]:
                    segs.append([p, q])
        self.lc = LineCollection(segs, linewidths=0.4, colors=(1, 1, 1, 0.05))
        self.ax_net.add_collection(self.lc)

        # neuron
        flat = np.array([p for layer in self.pos for p in layer])
        self.sc = self.ax_net.scatter(
            flat[:, 0], flat[:, 1], s=260, c=np.zeros(len(flat)),
            cmap="plasma", vmin=0, vmax=1,
            edgecolors="white", linewidths=0.8, zorder=3)

        for li, name in enumerate(self.LAYER_NAMES):
            self.ax_net.text(xs[li], 0.97, name, ha="center", va="top",
                             color="white", fontsize=9)
        for d, (x, y) in enumerate(self.pos[-1]):
            self.ax_net.text(x + 0.025, y, str(d), color="white",
                             va="center", fontsize=10)

        self.ring = self.ax_net.scatter([], [], s=520, facecolors="none",
                                        edgecolors="#00ff9c", linewidths=2.5, zorder=4)
        self.status = self.ax_net.text(0.5, 0.02, "Menyiapkan training...",
                                       ha="center", va="bottom",
                                       color="#9ad1ff", fontsize=11)
        self.ax_net.set_xlim(0, 1)
        self.ax_net.set_ylim(0, 1)
        self.ax_net.axis("off")

        # gambar sampel
        self.img = self.ax_img.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
        self.ax_img.axis("off")
        self.ax_img.set_title("Sampel", color="white", fontsize=10)

        # kurva akurasi & loss
        for ax, title in ((self.ax_acc, "Accuracy"), (self.ax_loss, "Loss")):
            ax.set_title(title, color="white", fontsize=10)
            ax.tick_params(colors="white", labelsize=8)
            for s in ax.spines.values():
                s.set_color("#444")
            ax.grid(alpha=0.2)
        (self.l_acc,) = self.ax_acc.plot([], [], "-o", color="#00e5ff", label="train", ms=3)
        (self.l_vacc,) = self.ax_acc.plot([], [], "-o", color="#ff9100", label="val", ms=3)
        (self.l_loss,) = self.ax_loss.plot([], [], "-o", color="#00e5ff", label="train", ms=3)
        (self.l_vloss,) = self.ax_loss.plot([], [], "-o", color="#ff9100", label="val", ms=3)
        self.ax_acc.legend(facecolor="#111", labelcolor="white", fontsize=8)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    # ---------- hitung "aktivitas" tiap layer ----------
    @staticmethod
    def _norm(v):
        v = np.abs(v)
        return v / (v.max() + 1e-8)

    def _levels(self, x, outs):
        img = x.reshape(28, 28)
        bands = np.array([b.mean() for b in np.array_split(img, self.NODES[0], axis=0)])
        levels = [bands / (bands.max() + 1e-8)]

        # 3 layer conv (output BatchNorm): rata-rata per channel
        for o, n in zip(outs[:3], self.NODES[1:4]):
            per_channel = np.abs(o[0]).mean(axis=(0, 1))
            idx = np.linspace(0, len(per_channel) - 1, n).astype(int)
            levels.append(self._norm(per_channel[idx]))

        # Dense 128
        d = outs[3][0]
        idx = np.linspace(0, len(d) - 1, self.NODES[4]).astype(int)
        levels.append(self._norm(d[idx]))

        # Output softmax (probabilitas langsung)
        levels.append(outs[4][0])
        return levels

    # ---------- update gambar ----------
    def _update(self, header):
        if not self.enabled:
            return
        i = self.k % len(self.x_samples)
        self.k += 1
        x = self.x_samples[i:i + 1]

        outs = [o.numpy() for o in self.probe(x, training=False)]
        levels = self._levels(x, outs)

        # warna neuron
        self.sc.set_array(np.concatenate(levels))

        # kekuatan koneksi = aktivitas neuron asal x neuron tujuan
        strengths = np.concatenate([
            np.outer(levels[li], levels[li + 1]).ravel()
            for li in range(len(levels) - 1)
        ])
        colors = plt.cm.plasma(np.clip(strengths, 0, 1))
        colors[:, 3] = 0.04 + 0.85 * np.clip(strengths, 0, 1)
        self.lc.set_color(colors)
        self.lc.set_linewidth(0.3 + 2.2 * np.clip(strengths, 0, 1))

        # tebak digit
        pred = int(np.argmax(outs[4][0]))
        self.ring.set_offsets([self.pos[-1][pred]])
        self.img.set_data(x.reshape(28, 28))
        self.ax_img.set_title(f"Asli: {int(self.y_samples[i])}  |  Tebakan: {pred}",
                              color="#00ff9c" if pred == int(self.y_samples[i]) else "#ff5252",
                              fontsize=10)
        self.status.set_text(header)
        self._redraw()

    def _redraw(self):
        if not self.enabled:
            return
        try:
            if not plt.fignum_exists(self.fig.number):
                self.enabled = False
                return
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            self.enabled = False

    def _update_curves(self):
        ep = range(1, len(self.hist["acc"]) + 1)
        for line, key in ((self.l_acc, "acc"), (self.l_vacc, "val_acc"),
                          (self.l_loss, "loss"), (self.l_vloss, "val_loss")):
            line.set_data(list(ep), self.hist[key])
        for ax in (self.ax_acc, self.ax_loss):
            ax.relim()
            ax.autoscale_view()

    # ---------- hook Keras ----------
    def on_train_begin(self, logs=None):
        bns = [l for l in self.model.layers if isinstance(l, layers.BatchNormalization)]
        dens = [l for l in self.model.layers if isinstance(l, layers.Dense)]
        # model "probe": hanya membaca output layer, tidak ikut training
        self.probe = tf.keras.Model(inputs=self.model.inputs,
                                    outputs=[l.output for l in bns + dens])
        self.total_epochs = (self.params or {}).get("epochs", "?")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.update_every == 0:
            logs = logs or {}
            self._update(
                f"Epoch {self.epoch}/{self.total_epochs}  |  batch {batch}  |  "
                f"loss {logs.get('loss', 0):.4f}  |  acc {logs.get('accuracy', 0) * 100:.2f}%")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.hist["acc"].append(logs.get("accuracy", 0))
        self.hist["val_acc"].append(logs.get("val_accuracy", 0))
        self.hist["loss"].append(logs.get("loss", 0))
        self.hist["val_loss"].append(logs.get("val_loss", 0))
        self._update_curves()
        self._update(
            f"Epoch {epoch + 1}/{self.total_epochs} selesai  |  "
            f"val_acc {logs.get('val_accuracy', 0) * 100:.2f}%  |  "
            f"val_loss {logs.get('val_loss', 0):.4f}")

    def finish(self):
        """Panggil di akhir script supaya jendela tetap terbuka."""
        plt.ioff()
        if self.enabled:
            plt.show()


# ============================================================
# KODE TRAINING (TIDAK DIUBAH)
# ============================================================
print("🚀 Starting high-accuracy CNN training for MNIST...")

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
aug.fit(x_train)

# CNN model architecture
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Visualizer (update_every = tampilan di-refresh tiap 50 batch;
# naikkan angkanya, misal 200, kalau ingin overhead lebih kecil)
viz = NeuralNetVisualizer(x_test[:300], y_test[:300], update_every=50)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    viz
]

# Training
start_time = time.time()
print("📚 Training CNN with augmentation...")
history = model.fit(
    aug.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=20,
    callbacks=callbacks,
    verbose=1
)
end_time = time.time()
print(f"⏱️ Training took {(end_time - start_time)/60:.2f} minutes")

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Save the model in Keras format
model.save('trained_digit_model_cnn.keras')
print("💾 CNN model saved as 'trained_digit_model_cnn.keras'")
print("🎉 Training complete! You can now run digit_recognition_app.py")

# Biarkan jendela visualisasi tetap terbuka (tutup jendela untuk keluar)
viz.finish()