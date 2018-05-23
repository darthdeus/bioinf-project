import tensorflow as tf
import numpy as np
import tensorflow.contrib.summary as tfsum
import datetime

class Autoencoder:
    def __init__(self):
        threads = 8
        graph = tf.Graph()
        self.session = tf.Session(graph=graph,
                                  config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                        intra_op_parallelism_threads=threads))

        with graph.as_default():
            self.images = tf.placeholder(tf.float32, shape=[None, 16, 16, 3], name="images")

            z_dim = 10

            def encoder(img):
                out = tf.layers.flatten(img)
                out = tf.layers.dense(out, 500, activation=tf.nn.relu)
                out = tf.layers.dense(out, 500, activation=tf.nn.relu)
                out = tf.layers.dense(out, z_dim, activation=tf.nn.relu)

                return out

            def decoder(z):
                out = tf.layers.dense(z, 500, activation=tf.nn.relu)
                out = tf.layers.dense(out, 500, activation=tf.nn.relu)
                out = tf.layers.dense(out, 16*16*3, activation=None)

                return tf.reshape(out, [-1, 16, 16, 3])

            self.z = encoder(self.images)
            self.generated_logits = decoder(self.z)
            self.generated_images = tf.nn.sigmoid(self.generated_logits, name="generated_images")

            self.loss = tf.losses.mean_squared_error(self.images, self.generated_images)

            global_step = tf.train.create_global_step()

            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step)

            logdir = "logs/autoencoder-{}-{}".format(z_dim, datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

            summary_writer = tfsum.create_file_writer(logdir, flush_millis=10*1000)
            with summary_writer.as_default(), tfsum.record_summaries_every_n_global_steps(100):
                self.summaries = [
                    tfsum.scalar("loss", self.loss),
                    tfsum.histogram("latent", self.z)
                ]

            self.generated_images_summary_data = tf.placeholder(tf.float32, [None, None, 3])
            with summary_writer.as_default(), tfsum.always_record_summaries():
                self.generated_images_summary = tfsum.image("generated_image", tf.expand_dims(self.generated_images_summary_data, axis=0))

            init = tf.global_variables_initializer()
            self.session.run(init)

            with summary_writer.as_default():
                tfsum.initialize(session=self.session, graph=self.session.graph)

    def train(self, images):
        _, _, loss, generated_images = self.session.run([self.training, self.summaries, self.loss, self.generated_images], {
            self.images: images
        })

        image_block = np.vstack([
            np.hstack([generated_images[i] for i in range(0,3)]),
            np.hstack([generated_images[i] for i in range(3,6)]),
            np.hstack([generated_images[i] for i in range(6,9)]),
            # np.hstack([generated_images[i] for i in range(12,16)]),
        ])

        self.session.run([self.generated_images_summary], {self.generated_images_summary_data: image_block})

        return loss


from glob import glob
import imageio

if __name__ == '__main__':
    X = np.stack([imageio.imread(img)[:, :, :3] for img in glob("downloads/*.png")], axis=0).astype(np.float32)
    X = X / 255.0

    net = Autoencoder()

    for i in range(2000):
        loss = net.train(X)

        if i % 100 == 0:
            print(loss)
