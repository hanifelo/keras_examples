#---------------------------------------------------------------

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x







def get_segnet(input_img,kernel_size, dropout, n_labels,batchnorm):
    c1 = conv2d_block(input_img, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c1 = Dropout(dropout*0.5)(c1)
    c2 = conv2d_block(c1, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c2= Dropout(dropout*0.5)(c2)
    c2= MaxPooling2D((2, 2))(c2)
# 
    c3 = conv2d_block(c2, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c3 = Dropout(dropout*0.5)(c3)
    c4 = conv2d_block(c3, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c4= Dropout(dropout*0.5)(c4)
    c4= MaxPooling2D((2, 2))(c4)
# 
    c5 = conv2d_block(c4, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c5 = Dropout(dropout*0.5)(c5)
    c6 = conv2d_block(c5, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c6= Dropout(dropout*0.5)(c6)
    c7 = conv2d_block(c6, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c7= Dropout(dropout*0.5)(c7)
    c7= MaxPooling2D((2, 2))(c7)
# 
    c8 = conv2d_block(c7, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c8 = Dropout(dropout*0.5)(c8)
    c9 = conv2d_block(c8, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c9= Dropout(dropout*0.5)(c9)
    c10 = conv2d_block(c9, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c10= Dropout(dropout*0.5)(c10)
    c10= MaxPooling2D((2, 2))(c10)
# 
    c11 = conv2d_block(c10, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c11 = Dropout(dropout*0.5)(c11)
    c12 = conv2d_block(c11, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c12= Dropout(dropout*0.5)(c12)
    c13 = conv2d_block(c12, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c13= Dropout(dropout*0.5)(c13)
    c13= MaxPooling2D((2, 2))(c13)
# 
   #decoding_layers  
    p1= UpSampling2D(size=(2,2))(c13)
    # 
    c14 = conv2d_block(p1, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c14 = Dropout(dropout*0.5)(c14)
    c15 = conv2d_block(c14, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c15= Dropout(dropout*0.5)(c15)
    c16 = conv2d_block(c15, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c16= Dropout(dropout*0.5)(c16)
    
    
    p2= UpSampling2D(size=(2,2))(c16)
    # 
    c17 = conv2d_block(p2, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c17 = Dropout(dropout*0.5)(c17)
    c18 = conv2d_block(c17, n_filters=512, kernel_size=3, batchnorm=batchnorm)
    c18= Dropout(dropout*0.5)(c18)
    c19 = conv2d_block(c18, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c19= Dropout(dropout*0.5)(c19)
    
    
    p3= UpSampling2D(size=(2,2))(c19)
    # 
    c20 = conv2d_block(p3, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c20 = Dropout(dropout*0.5)(c20)
    c21 = conv2d_block(c20, n_filters=256, kernel_size=3, batchnorm=batchnorm)
    c21= Dropout(dropout*0.5)(c21)
    c22 = conv2d_block(c21, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c22= Dropout(dropout*0.5)(c22)
   
    
    p4= UpSampling2D(size=(2,2))(c22)
    # 
    c23 = conv2d_block(p4, n_filters=128, kernel_size=3, batchnorm=batchnorm)
    c23 = Dropout(dropout*0.5)(c23)
    c24 = conv2d_block(c23, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c24= Dropout(dropout*0.5)(c24)
    
    p5= UpSampling2D(size=(2,2))(c24)
    # 
    c25 = conv2d_block(p5, n_filters=64, kernel_size=3, batchnorm=batchnorm)
    c25 = Dropout(dropout*0.5)(c25)

    #end layer number of labels
    c26 = Conv2D(filters=n_labels, kernel_size=(1, 1), kernel_initializer="he_normal",
               padding="same")(c25)
    outputs=BatchNormalization()(c26)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


input_img = Input((im_height, im_width, 1), name='img')

model = get_segnet(input_img,kernel_size=3,  dropout=0.05, n_labels=1,batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()





