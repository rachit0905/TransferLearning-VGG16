from keras.applications import VGG16
from keras.layers import Flatten,Dense,Dropout
from keras.models import Model

img_rows=224
img_cols=224

##224 is taken,as the VGG16 was designed to use on 224x224 pixels

## VGG is imported using the weights of imagenet and without the including of the top layer. 
vgg16=VGG16(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))


for layer in vgg16.layers:
    layer.trainable=False
    
for(i, layer) in enumerate(vgg16.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


def addTopModel(bottom_model,num_classes,D=256):
    top_model=bottom_model.output
    top_model=Flatten(name="flattern")(top_model)
    top_model=Dense(D,activation='relu')(top_model)
    top_model=Dropout(0.3)(top_model)
    top_model=Dense(num_classes,activation='softmax')(top_model)
    return top_model

num_classes = 17
FC_Head=addTopModel(vgg16,num_classes)

model=Model(inputs=vgg16.input,outputs=FC_Head)

print(model.summary())

from keras.preprocessing.image import ImageDataGenerator


#data augumentation

train_data_dir = './train'
validation_data_dir = './validation'
train_datagen = ImageDataGenerator(
      rescale=1./255,rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
 
train_batchsize=16
val_batchsize=10 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=val_batchsize,
        class_mode='categorical',shuffle=False)

##Training the model 
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint,EarlyStopping

checkpoint=ModelCheckpoint("Flowers/flower.h5",monitor='val_loss',mode='min',verbose=1,
                           save_best_only=True)

earlystop=EarlyStopping(monitor='val_loss',mode='min',patience=3,restore_best_weights=True,
                        min_delta=0,
                        verbose=1)

callbacks=[checkpoint,earlystop]


model.compile(loss="categorical_crossentropy",optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])


nb_train_samples=1190
nb_validation_samples=170



epochs=3
batch_size=16

history=model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_samples,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples)
































 



    