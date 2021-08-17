'''
Created on Apr 10, 2020

@author: Sagar Kale
'''
import caffe


# To create caffemodel for this prototxt
net_new = caffe.Net("deploy_new.prototxt",caffe.TEST)


# To read prototxt and caffemodel file from which weights need to copy
net_old = caffe.Net("deploy_old.prototxt", "deploy_old.caffemodel",caffe.TEST)

# Print all layers with/without parameters
for new_key in net_new.blobs.keys():
    print('New Model Layer', new_key)
    
# Print all layers with/without parameters    
for old_key in net_old.blobs.keys():
    print('Old Model Layer', new_key)  

#Only print layers with parameters
for new_key, old_key in zip(net_new.params.keys(), net_old.params.keys()):
    print('Layers with weights', new_key, old_key)

#Copy old trained weights into new model 
for old_key in net_old.params.keys():
    #Weights
    net_new.params[old_key][0].data[...] = net_old.params[old_key][0].data[...]
    #Bias
    net_new.params[old_key][1].data[...] = net_old.params[old_key][1].data[...]
    print("Weights copied succesfully for ", old_key)


# To save the caffemodel
net_new.save("deploy_new.caffemodel")
