# In this code: 
# 1) we can convert keras DNN model('.h5' files) with 2 hidden layers
#    to numpy array (or '.csv' files)
# 2) we cannot deal with batch normalization

# -------------------------------------------

# file path of the h5 file that you want to read
filename = 'model.h5'

# file path of csv file you want to output
csvname = 'out.csv'

# -------------------------------------------
# check the constructure of h5 file
# -------------------------------------------
import h5py    
import numpy as np
np.set_printoptions(threshold=np.inf)
f1 = h5py.File(filename,'r+') 

## Check the group name of hdf5 directory
# weights matrices are saved in 'kernel:0' in 'dense_XX' in 'model_weights' group.
# bias vectors are saved in 'bias:0' in 'dense_XX' in 'model_weights' group.
# 'XX' represents the XXth layer.

# < (1) first group directory name>
# The first group directory of the file
# for example, this is 'model_weights'
# If you find 'model_weights', set 'model_weights' as the first group directory name.
a_group_key1 = list(f1.keys())[0]
print(a_group_key1)

# < (2) second group directory name>
# The awcond group directory of the file
# for example, this is 'dense_XX' or 'activation_XX' (XX represents the XXth layer)
# If you want to know the first weight matrix, set 'dense_1'
# as the second group directory name. 
a_group_key2 = list(f1[a_group_key1])
print(a_group_key2)

# < (3) repeat (2) until you get 'kernel:0' or 'bias:0'>
# for example, when you get 'dense_1' at (2), run the code below:
a_group_key3 = list(f1[a_group_key1 + '/dense_1']) 
print(a_group_key3)
# repeat this until you get 'kernel:0' or 'bias:0'.

# < (4) last group directory name>
# Lastely you get 'kernel:0' or 'bias:0'.
# If you want to get the weight matrix, choose 'kernel:0' and 
# if you want to get the bias vector, choose 'bias:0'.


# By doing (1) to (4), you will get the full address.
# for example, you may get the below address when you want to know 
# about the first weight matrix:
# 'model_weights/dense_1/dense_1/kernel:0'
# Be careful that this is just an example and maybe you get 
# 'model_weights/dense_51/dense_51/kernel:0' or 
# 'model_weights/dense_1/dense_1/dense_1/kernel:0' or others
# the address may change a little by the situation when the h5 file was saved.

# set the address you get in above
address = 'model_weights/dense_1/dense_1/kernel:0'

# -------------------------------------------
# read the h5 file and convert it to numpy array and csv
# -------------------------------------------
# convert h5 data to numpy array(float32)
values = np.array(f1[address]).astype("float32")

# output numpy array as csv file
np.savetxt(csvname, values, delimiter=",")

print('OK')