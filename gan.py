import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.dense1 = tf.keras.layers.Dense(17*17*256, use_bias=False,input_shape=(277,))
        self.activ1 = tf.keras.layers.LeakyReLU()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.reshap1 = tf.keras.layers.Reshape((17,17,256))

        self.transcov2 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(3, 3), padding='same', use_bias=False)
        self.activ2 = tf.keras.layers.LeakyReLU()
        self.norm2 = tf.keras.layers.BatchNormalization()
        
        self.transcov3 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(5, 5), padding='same',use_bias=False)
        self.activ3 = tf.keras.layers.LeakyReLU()
        self.norm3 = tf.keras.layers.BatchNormalization()
        
        self.transcov4 = tf.keras.layers.Conv2DTranspose(3, (2, 2), strides=(1, 1), padding='same', use_bias=False)
        
    def call(self, x, training=True):
        x = self.dense1(x)
        x = self.norm1(x, training=training)
        x = self.activ1(x)
        x = self.reshap1(x)
        
        x = self.transcov2(x)
        x = self.norm2(x, training=training)
        x = self.activ2(x)
        
        x = self.transcov3(x)
        x = self.norm3(x, training=training)
        x = self.activ3(x)
           
        x = self.transcov4(x)
        
        x = tf.nn.tanh(x)
        
        return x
    
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.cov1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same',input_shape=[255, 255, 3])
        self.activ1 = tf.keras.layers.LeakyReLU()
        self.drop1 = tf.keras.layers.Dropout(0.3)
        self.maxpool1 = tf.keras.layers.MaxPooling2D()
        
        self.cov2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')
        self.activ2 = tf.keras.layers.LeakyReLU()
        self.drop2 = tf.keras.layers.Dropout(0.3)
        self.maxpool2 = tf.keras.layers.MaxPooling2D()
        
        self.cov3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')
        self.activ3 = tf.keras.layers.LeakyReLU()
        self.drop3 = tf.keras.layers.Dropout(0.3)
        self.maxpool3 = tf.keras.layers.MaxPooling2D()
        
        self.flatten4 = tf.keras.layers.Flatten()
        self.dense4 = tf.keras.layers.Dense(1024)
        self.Dense = tf.keras.layers.Dense(1)
        
    def call(self, x, training=True):
        x = self.cov1(x)
        x = self.activ1(x)
        x = self.maxpool1(x)
        x = self.drop1(x)
        
        x = self.cov2(x)
        x = self.activ2(x)
        x = self.maxpool2(x)
        x = self.drop2(x)
        
        x = self.cov3(x)
        x = self.activ3(x)
        x = self.maxpool3(x)
        x = self.drop3(x)
        
        x = self.flatten4(x)
        x = self.dense4(x)
        
        #mid = x
        
        D = self.Dense(x)   
        
        return D, x
    
class QNet(tf.keras.Model):
    def __init__(self):
        super(QNet, self).__init__()
        
        self.Qd = tf.keras.layers.Dense(128)
        self.Qb = tf.keras.layers.BatchNormalization()
        self.Qa = tf.keras.layers.LeakyReLU()
        
        self.Q_cat = tf.keras.layers.Dense(7)
        self.Q_con1_mu = tf.keras.layers.Dense(2)
        self.Q_con1_var = tf.keras.layers.Dense(2)
        self.Q_con2_mu = tf.keras.layers.Dense(2)
        self.Q_con2_var = tf.keras.layers.Dense(2)
        
    def sample(self, mu, var):
        eps = tf.random.normal(shape=mu.shape)
        sigma = tf.sqrt(var)
        z = mu + sigma * eps
        
        return z
    
    def call(self, x, training=True):
        x = self.Qd(x)
        x = self.Qb(x, training=training)
        x = self.Qa(x)
        
        q = x
        
        Q_cat = self.Q_cat(q)

        Q_con1_mu = self.Q_con1_mu(q)
        Q_con1_var = tf.exp(self.Q_con1_var(q))
        Q_con2_mu = self.Q_con2_mu(q)
        Q_con2_var = tf.exp(self.Q_con2_var(q))
        
        Q_con1 = self.sample(Q_con1_mu, Q_con1_var)
        Q_con2 = self.sample(Q_con2_mu, Q_con2_var)
        
        return Q_cat, Q_con1, Q_con2

        
