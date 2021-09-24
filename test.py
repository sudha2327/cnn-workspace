from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

json_file=open('model.json','r')
loaded_model_json=json_file.read()
#for closing

json_file.close()

#now loade the model weights file

model=model_from_json(loaded_model_json)
model.load_weights('model.h5')
print("model has been successfully loaded")

def classify(img_file):
    img_name=img_file
    test_img=image.load_img(img_name,target_size=(64,64))

    test_img=image.img_to_array(test_img)
    test_img=np.expand_dims(test_img,axis=0)
    result=model.predict(test_img)

    if result[0][0]==1:
        prediction='barbie'
    else:
        prediction='joker'

    print(prediction,img_name)

#in above code we have done that prediction
#now we will get into that os path codes

import  os

path="S:/coding/Intern project/CNN workplace/Dataset/test"

files=[]

for r , d, f in os.walk(path):
    for file in f:
        if '.jbeg' in file:
            files.append(os.path.join(r,file))

for f in files:
    classify(f)
    print('\n')





