from genderclassifierfunctions import unzipfile
from genderclassifierfunctions import loadjsonfiletolist
from genderclassifierfunctions import writecsvfile
from genderclassifierfunctions import clean
from genderclassifierfunctions import feature_processing
from genderclassifierfunctions import feature_selection_classifier_1
from genderclassifierfunctions import feature_selection_classifier_2
from genderclassifierfunctions import feature_selection_classifier_3
from genderclassifierfunctions import kmeans_classification
from genderclassifierfunctions import minibatchkmeans_classifier
from genderclassifierfunctions import plot3d
from genderclassifierfunctions import plot2d
from genderclassifierfunctions import eigen_decomposition
from genderclassifierfunctions import concatenate_result
from sklearn.decomposition import PCA
import genderclassifierfunctions

# unzipfile('test_data.zip', 'welcometotheiconic')

# Also removed duplication
keys, datalist = loadjsonfiletolist('data.json')

# Also converted to numpy array
dataarray = clean(datalist)

# Also normalized
allfeatures = feature_processing(dataarray)

# Classifier 1 - K means
featureClassifier1 = feature_selection_classifier_1(allfeatures)
result_1_kmeans = kmeans_classification(2, featureClassifier1)

# Classifier 2 - K means
featureClassifier2 = feature_selection_classifier_2(allfeatures)
result_2_kmeans = kmeans_classification(2, featureClassifier2)

# Classifier 3 - Mini Batch K means with PCA on all features
featureClassifier3 = feature_selection_classifier_3(allfeatures)
pcaResult = PCA(n_components=7).fit(featureClassifier3).transform(featureClassifier3)
result_3_mbk = minibatchkmeans_classifier(2, featureClassifier3)

# Unsupervised ensemble learning, final label needs to get at least 2 votes from 3 classifiers
# female = 0, male = 1
final_result = (result_1_kmeans + result_1_kmeans + result_3_mbk)
final_result = [1 if row > 1 else 0 for row in final_result]
print("Total male:", sum(final_result))
print("First 30 final results female label = 0/ male label =1")
print(*final_result[:30])

# Print result to CSV file for easy read
resultkeys, resultdata = concatenate_result(keys, datalist, "gender(female=0)", final_result)
writecsvfile('GenderLabellingFinal.csv', resultkeys, resultdata)

# Plot final result in 2D with first 2 components
plot2d(pcaResult, final_result, 0, 1)

# Plot in 3D with first 3 components
plot3d(pcaResult, final_result, 0, 1, 2)

# The step is not required, I use eigendecomposition to find which features contribute most to the cluster result
# This helps to select the features and understand the result
eigen_decomposition(featureClassifier1, range(featureClassifier1.shape[1]))

# Appendix: Python Documentation
# For the detailed documentation of functions in file _genderclassifierfunctions.py_ please run
help(genderclassifierfunctions)
