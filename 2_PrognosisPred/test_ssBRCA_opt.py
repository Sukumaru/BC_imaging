#works for 

# pandas==0.25.1
# numpy==1.17.2
# scikit-learn==0.21.3
# Theano==1.0.4
# tqdm==4.36.1
# dill==0.3.3.1
####
#Code tested on my machine;

######################
from cox_nnet_v2 import *
import numpy
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import random
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from tqdm import tqdm

numpy.random.seed(42)
random.seed(42)

#setting seeds for reproducibity
#os.environ['PYTHONHASHSEED']=str(42)

d_path = "BRCA/"


############# data loading#####

x1 = numpy.loadtxt(fname=d_path+"259p_clinical4.csv",delimiter=",",skiprows=1)
x2 = numpy.loadtxt(fname=d_path+"259p_phenotypic27.csv",delimiter=",",skiprows=1)
x3 = numpy.loadtxt(fname=d_path+"259p_micro273.csv",delimiter=",",skiprows=1)
x4 = numpy.loadtxt(fname=d_path+"259p_tumor105.csv",delimiter=",",skiprows=1)

################################################################################# 2stage data loader; no header names
#x2 = numpy.loadtxt(fname="ht_259p_phenotypic27.csv",delimiter=",",skiprows=0)
#x3 = numpy.loadtxt(fname="ht_259p_micro273.csv",delimiter=",",skiprows=0)
#x4 = numpy.loadtxt(fname="ht_259p_tumor105.csv",delimiter=",",skiprows=0)


###############################################################################

#print(x1.shape,x2.shape,x3.shape,x4.shape)
#print(x2.shape,x3.shape,x4.shape)

#x = numpy.concatenate((x2,x3,x4), axis=1)

x = x2

x = x[:, (x != 0).any(axis=0)] # removes features which are all zero

print (x.shape)

scaler = MinMaxScaler()
x_all = scaler.fit_transform(x)
ytime = numpy.loadtxt(fname=d_path+"ytime.csv",delimiter=",",skiprows=0)
ystatus = numpy.loadtxt(fname=d_path+"ystatus.csv",delimiter=",",skiprows=0)


###
###
print(x_all.shape,ytime.reshape(-1,1).shape,ystatus.reshape(-1,1).shape)


############## with L2 search and model train ########################################################################


# # split into test/train sets

# x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
#     sklearn.model_selection.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 1)

# # split training into optimization and validation sets
# x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
#     sklearn.model_selection.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 1)

# # # set parameters
# model_params = dict(node_map = None, input_split = None)
# # #search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
# # #    max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
# # #    rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0) ### for nesterov accelerated coxnnetv1
# search_params = dict(method = "adam", learning_rate=0.001, beta1=0.9, beta2=0.999,epsilon=1e-8, 
#     max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
#     rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)




# # ############################ regularixation parameter search
# cv_params = dict(cv_metric = "cindex", L2_range = numpy.arange(-6,1,0.20))

# #print ('Finding L2 parameter')
# print ('Finding L2 parameter')
# #profile log likelihood to determine lambda parameter
# likelihoods, L2_reg_params = L2Profile(x_opt,ytime_opt,ystatus_opt,
#     x_validation,ytime_validation,ystatus_validation,
#     model_params, search_params, cv_params, verbose=False)

# print('likelihoods:',likelihoods)


# ##build model based on optimal lambda parameter
# print('argmax_index', numpy.argmax(likelihoods))
# L2_reg = L2_reg_params[numpy.argmax(likelihoods)]
# # #############################################################################


# ###############################################################################
# #L2_reg = -2.25
# print('best L2 before exp is : ', L2_reg)
# print('best L2 value is: ', numpy.exp(L2_reg))
# model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
# print ('Training Cox NN')

# #start = time.time()

# model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)

# # theta_all = model.predictNewData(x)

# #print (theta_all)

# cindex_val = CIndex(model, x_test, ytime_test, ystatus_test)

# print ('c-index',cindex_val)

#end = time.time()

#print ('Total time in seconds', end-start)

#numpy.savetxt(d_path+"BRCA_theta.csv", theta_all, delimiter=",")
#numpy.savetxt(d_path+"LUAD_ytime_test.csv", ytime_test, delimiter=",")
#numpy.savetxt(d_path+"LUAD_ystatus_test.csv", ystatus_test, delimiter=",")


# #################### model save #########################

#saveModel(model, 'BRCA_cindex_'+str(round(cindex_val,3))+'.pkl')
#print('model save works')
# ########## var importance without model saving############

# print('Running relative variable importance')

# varimp = varImportance(model, x_train, ytime_train, ystatus_train)

# result = varimp/max(varimp)

# print(list(numpy.around(numpy.array(result),3)))

# #print('variable importance finished!')

# ########## feature importance fischer 2018 ############

# print('Running feature importance fischer 2018')

# feaimp = permutationImportance(model, 1, x_train, ytime_train, ystatus_train)

# result = feaimp/max(feaimp)

# print(list(numpy.around(numpy.array(result),3)))

# print('Running sign of beta')

# sigbeta = signOfBeta(model, x_test)

# print(sigbeta)

# print('Finished!')
############################################# multiple BATCH RUNs bELOW

l2_opti_train = {}
l2_opti_test = {}
l2_opti_train_std = {}
l2_opti_test_std = {}


#for l2_val in tqdm(numpy.arange(-8,-1,0.25)):
for l2_val in tqdm(numpy.arange(-4.25,-3.75,0.025)):

	list_of_cindex_train = []
	list_of_cindex_test = []

	for i in range(0,10):

		x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = train_test_split(x_all, ytime, ystatus, test_size = 0.15, random_state = i)

		# split training into optimization and validation sets
		# x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
		    # sklearn.model_selection.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 1)

		# set parameters
		model_params = dict(node_map = None, input_split = None)
		# search_params = dict(method = "nesterov", learning_rate=0.001, momentum=0.9, 
		#     max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
		#     rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0) ### for nesterov accelerated coxnnetv1
		search_params = dict(method = "adam", learning_rate=0.1, beta1=0.9, beta2=0.999,epsilon=1e-8, 
		    max_iter=10000, stop_threshold=0.995, patience=2000, patience_incr=2, 
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
		list_of_cindex_train.append(round(CIndex(model, x_train, ytime_train, ystatus_train),4))
		list_of_cindex_test.append(round(CIndex(model, x_test, ytime_test, ystatus_test),4))
		#print ('random_state_', i ,' c-index ', CIndex(model, x_test, ytime_test, ystatus_test))

	#print (l2_val)
	#print ('list_of_cindex_train',list_of_cindex_train)
	#print ('mean_cindex_train', numpy.asarray(list_of_cindex_train).mean())
	#print ('std_cindex_train', numpy.asarray(list_of_cindex_train).std())

	#print ('list_of_cindex_test',list_of_cindex_test)
	#print ('mean_cindex_test', numpy.asarray(list_of_cindex_test).mean())
	#print ('std_cindex_test', numpy.asarray(list_of_cindex_test).std())

	l2_opti_train[round(l2_val,4)] = round(numpy.asarray(list_of_cindex_train).mean(),4)
	l2_opti_test[round(l2_val,4)]  = round(numpy.asarray(list_of_cindex_test).mean(),4)
	l2_opti_train_std[round(l2_val,4)] = round(numpy.asarray(list_of_cindex_train).std(),4)
	l2_opti_test_std[round(l2_val,4)]  = round(numpy.asarray(list_of_cindex_test).std(),4)

print ('train_means',l2_opti_train)
print ('test_means',l2_opti_test)
print ('train_std',l2_opti_train_std)
print ('test_std',l2_opti_test_std)
# ##################################################################
