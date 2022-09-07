#works for 

# pandas==0.25.1
# numpy==1.17.2
# scikit-learn==0.21.3
# Theano==1.0.4
# tqdm==4.36.1
# pickle==4.0
####
#Code tested on my machine;

######################
from cox_nnet_v2 import *
import numpy
import sklearn
import sklearn.model_selection
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

###################################################################################### input hidden layer values


d_path = "BRCA/"

############# data loading#####

#x1 = numpy.loadtxt(fname=d_path+"259p_clinical4.csv",delimiter=",",skiprows=1)
x2 = numpy.loadtxt(fname="ht_259p_phenotypic27.csv",delimiter=",",skiprows=0)
x3 = numpy.loadtxt(fname= "ht_259p_micro273.csv",delimiter=",",skiprows=0)
x4 = numpy.loadtxt(fname= "ht_259p_tumor105.csv",delimiter=",",skiprows=0)

#print(x2.shape)
print(x2.shape,x3.shape,x4.shape)

x = numpy.concatenate((x3,x4), axis=1)
#x = x2

print (x.shape)


#
scaler  = MinMaxScaler() #works best
x_hidden   = scaler.fit_transform(x)



ytime   = numpy.loadtxt(fname=d_path+"ytime.csv",delimiter=",",skiprows=0)
ystatus = numpy.loadtxt(fname=d_path+"ystatus.csv",delimiter=",",skiprows=0)
#
#
#
#
#x_hidden = x_all   #x_all
# # split into test/train sets

# x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
#     sklearn.model_selection.train_test_split(x_hidden, ytime, ystatus, test_size = 0.2, random_state = 1)

# # split training into optimization and validation sets
# # x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
# #     sklearn.model_selection.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 1)

# # set parameters
# model_params = dict(node_map = None, input_split = None)
# # search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
# #     max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
# #     rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0) ### for nesterov accelerated coxnnetv1
# search_params = dict(method = "adam", learning_rate=0.0001, beta1=0.9, beta2=0.999,epsilon=1e-8, 
#     max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
#     rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)




# # ############################ regularixation parameter search
# # cv_params = dict(cv_metric = "cindex", L2_range = numpy.arange(-6,0,0.5))

# # #print ('Finding L2 parameter')
# # print ('Finding L2 parameter')
# # #profile log likelihood to determine lambda parameter
# # likelihoods, L2_reg_params = L2Profile(x_opt,ytime_opt,ystatus_opt,
# #     x_validation,ytime_validation,ystatus_validation,
# #     model_params, search_params, cv_params, verbose=False)

# # print('likelihoods:',likelihoods)

# # ##build model based on optimal lambda parameter
# # L2_reg = L2_reg_params[numpy.argmax(likelihoods)]
# #############################################################################
# #manual tuning may be required sometimes
# L2_reg = -4
# ############################################################################

# print('best L2 before exp is : ', L2_reg)
# print('best L2 value is: ', numpy.exp(L2_reg))
# model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
# print ('Training Cox NN')

# #start = time.time()

# model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)

# #theta = model.predictNewData(x_test)
# cindex_train = CIndex(model, x_train, ytime_train, ystatus_train)
# cindex_test = CIndex(model, x_test, ytime_test, ystatus_test)

# print ('c-index_train',cindex_train)

# print ('c-index_test',cindex_test)

#end = time.time()

#print ('Total time in seconds', end-start)
############################# importance of hidden layers

# print('Running feature importance fischer 2018')

# feaimp = permutationImportance(model, 20, x_hidden, ytime, ystatus)

# #result_feaimp = feaimp/max(feaimp)

# feaimp = list(numpy.around(numpy.array(feaimp),3))

# print(feaimp)

# print('featImp 2018 importance finished!')


l2_opti_train = {}
l2_opti_test = {}
l2_opti_train_std = {}
l2_opti_test_std = {}


for l2_val in tqdm(numpy.arange(-10,0,0.5)):
#for l2_val in tqdm(numpy.arange(-6.50,-5.5,0.05)):

	list_of_cindex_train = []
	list_of_cindex_test = []

	for i in range(0,10):

		x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
		    sklearn.model_selection.train_test_split(x_hidden, ytime, ystatus, test_size = 0.2, random_state = i)

		# split training into optimization and validation sets
		# x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
		    # sklearn.model_selection.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 1)

		# set parameters
		model_params = dict(node_map = None, input_split = None)
		# search_params = dict(method = "nesterov", learning_rate=0.001, momentum=0.9, 
		#     max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
		#     rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0) ### for nesterov accelerated coxnnetv1
		search_params = dict(method = "adam", learning_rate=0.001, beta1=0.9, beta2=0.999,epsilon=1e-8, 
		    max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
		    rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)


		# cv_params = dict(cv_metric = "cindex", L2_range = numpy.arange(-4,2,0.25))

		# #print ('Finding L2 parameter')
		# print ('Finding L2 parameter')
		# #profile log likelihood to determine lambda parameter
		# likelihoods, L2_reg_params = L2Profile(x_opt,ytime_opt,ystatus_opt,x_validation,ytime_validation,ystatus_validation,model_params, search_params, cv_params, verbose=False)



		# ####build model based on optimal lambda parameter
		# L2_reg = L2_reg_params[numpy.argmax(likelihoods)] #-1.75 best for integrated, -4 for inter, 
		#L2_reg = -3.8
		L2_reg = l2_val
		#print ('best L2 value before exp', L2_reg)


		#print('best L2 value is: ', numpy.exp(L2_reg))
		model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
		#print ('Training Cox NN')
		model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)

		#theta = model.predictNewData(x_test)

		#print ('c_index: ',  concordance_index(ytime_test, theta, ystatus_test))
		list_of_cindex_train.append(round(CIndex(model, x_train, ytime_train, ystatus_train),3))
		list_of_cindex_test.append(round(CIndex(model, x_test, ytime_test, ystatus_test),3))
		#print ('random_state_', i ,' c-index ', CIndex(model, x_test, ytime_test, ystatus_test))

	#print (l2_val)
	#print ('list_of_cindex_train',list_of_cindex_train)
	#print ('mean_cindex_train', numpy.asarray(list_of_cindex_train).mean())
	#print ('std_cindex_train', numpy.asarray(list_of_cindex_train).std())

	#print ('list_of_cindex_test',list_of_cindex_test)
	#print ('mean_cindex_test', numpy.asarray(list_of_cindex_test).mean())
	#print ('std_cindex_test', numpy.asarray(list_of_cindex_test).std())

	l2_opti_train[l2_val] = numpy.asarray(list_of_cindex_train).mean()
	l2_opti_test[l2_val]  = numpy.asarray(list_of_cindex_test).mean()
	l2_opti_train_std[l2_val] = numpy.asarray(list_of_cindex_train).std()
	l2_opti_test_std[l2_val]  = numpy.asarray(list_of_cindex_test).std()

print ('train_means',l2_opti_train)
print ('test_means',l2_opti_test)
print ('train_std',l2_opti_train_std)
print ('test_std',l2_opti_test_std)
##################################################################




