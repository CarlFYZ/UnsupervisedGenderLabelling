import json
import csv
import numpy as np
from zipfile import ZipFile
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import pylab as pl
import scipy as sp
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans


def unzipfile(filename, passcode):
    """
    Unzip the file specified with password calculated from passcode
    :param filename: file name to be unzipped
    :param passcode: passcode used to calculate the password of the zip file which is "an unserialized SHA-256 hash" of the passwcode
    """
    # Password is SHA-256 hash of the pass code received
    password = hashlib.sha256(passcode.encode('utf-8')).hexdigest()
    # Unzip with password
    with ZipFile(filename) as zf:
        zf.extractall(pwd=bytes(password, 'utf-8'))


def loadjsonfiletolist(filename):
    """
    Load json file as python list
    :param filename: the json file name to be loaded
    :return: the column name and data
    """
    with open(filename) as json_file:
        json_data = json.load(json_file)

    # Convert data to a list of list
    datalist = []
    # Get the keys
    json_row_1 = json_data[0]
    # Save customer_ids for duplication check
    customer_ids = set();
    # Iterate through json_data to build list of list first
    count = 0
    for row in json_data:
        # Remove duplication
        customer_id = row.get('customer_id')
        if customer_id in customer_ids:
            # print("skip", customer_id)
            count += 1
            continue
        else:
            customer_ids.add(customer_id)

        # Fill inner list
        oneCustomer = []
        for key in json_row_1.keys():
            value = row.get(key)
            if value is None:
                # print("Missing", key)
                oneCustomer.append(0)
            else:
                if "N" == value:
                    oneCustomer.append('0')
                elif "Y" == value:
                    oneCustomer.append('1')
                else:
                    oneCustomer.append(value)
        datalist.append(oneCustomer)
    print("Duplication:", count)
    return json_row_1.keys(), datalist


def concatenate_result(keys, datalist, resultkey, resultdata):
    """
    Append the result (gender labels) as the last column to key and value for saving final result in file

    :param keys: original keys of table
    :param datalist: original data
    :param resultkey: name of result column "gender"
    :param resultdata: value of result, i.e. all the gender labels as column
    """
    # append result row by row
    i = 0
    for row in datalist:
        row.append(resultdata[i])
        i += 1
    # append column name
    newkeys = list(keys)
    newkeys.append(resultkey)
    print("Display first 5 classification result: customer_id, gender(femal=0)")
    for row in datalist[:5]:
        print(row[0], row[-1])
    return newkeys, datalist


def writecsvfile(filename, columnnames, data):
    """
    Save the list of list in csv file for visualization
    :param filename: name of targe file
    :param columnnames: column names to be saved
    :param data: data to be saved
    """
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(columnnames)  # header row
        for row in data:
            writer.writerow(row[:])


def clean(listoflist):
    """
    Following corrections are made to the data
        days_since_last_order is in unit of hours, divide it by 24 to get the correctvalue
        average_discoun\t_used is multipliedby 1000, divideby 1000 to get the correctvalue
        Removed 249 duplicated records by checking key customer_id
        Field coupon_discount_applied is missing for some records causing data not aligned, set 0 if missing to fix the issue
        Field is_newsletter_subscriber has string value Y/N, converted to 1/0 for further processing
        returns is rutruned items not returnd orders
    Found following issues/finding in data but no correction is made because not necessary or lack of information
        items in subcategories like wapp_items, wspt_items etc don't add up to the total items, which means some discrepancy in data
        different_addresses is a boolean value Ture/False, it probably shows if billing and shipping address are different,
        it doesn't match the discription in github "Number of times a different billing and shipping address was used" https://github.com/theiconic/datascientist
        Same boolean vs integer issue can be found with value cc_payments, paypal_payments, afterpay_payments, apple_payments.
        Boolean features are normally not used by k-means clustering
        Extra columns redpen_discount_used, coupon_discount_applied are available
        Few customers apprently were buying for resale, their buying pattern may be different
    :param listoflist: list to be converted to 2d numpy array
    :return: 2d numpy array
    """
    # Firs column is Customer ID, not used for this classification, remove
    array2d = np.array(listoflist)
    array2d = np.delete(array2d, 0, 1)

    # convert array from string to float
    array2d = array2d.astype(np.float)

    # days_since_last_order is in hours, divide by 24 to get the correct value
    array2d[:, 1] = array2d[:, 1] / 24

    # average_discount_used is multiplied by 1000, divide by 1000 to get the correct value, keep 4 decimal places
    array2d[:, 40] = np.around(array2d[:, 40] / 10000, 4)

    return array2d


def feature_processing(array2d):
    """
    Extract and prepare all the needed features
    Also normalize each feature because K means is sensitive to variance in data
    Categorical values are skipped for K means doesn't work well with them due to only 2 values 0/1
    Absolute values are not used, they are more related with how long the customer is with ICONIC and less related with
    gender, hence they may do more harm than good
    I believe Curvy item has a strong correlation with gender, however they are very right-skewed use np.power(1/6) to
    smooth it
    :param array2d: the 2D array of data [ num_samples * num_features ]
    :return: the array with extracted features, 29 features are extracted as following
    """
    new_array2d = np.zeros([array2d.shape[0], 29])
    # items/ orders
    new_array2d[:, 0] = array2d[:, 4] / array2d[:, 3]
    # cancels / orders
    new_array2d[:, 1] = array2d[:, 5] / array2d[:, 3]
    # returns / items
    new_array2d[:, 2] = array2d[:, 6] / array2d[:, 4]
    # voucher / orders
    new_array2d[:, 3] = array2d[:, 10] / array2d[:, 3]
    # female_items / female_items + male_items
    new_array2d[:, 4] = array2d[:, 15] / ([1 if x == 0 else x for x in (array2d[:, 15] + array2d[:, 16])])
    # male_items / female_items + male_items
    new_array2d[:, 5] = array2d[:, 16] / ([1 if x == 0 else x for x in (array2d[:, 15] + array2d[:, 16])])
    # unisex_items / items
    new_array2d[:, 6] = array2d[:, 17] / array2d[:, 4]
    # wapp_items / items
    new_array2d[:, 7] = array2d[:, 18] / array2d[:, 4]
    # wftw_items / items
    new_array2d[:, 8] = array2d[:, 19] / array2d[:, 4]
    # mapp_items / items
    new_array2d[:, 9] = array2d[:, 20] / array2d[:, 4]
    # wacc_items / items
    new_array2d[:, 10] = array2d[:, 21] / array2d[:, 4]
    # macc_items / items
    new_array2d[:, 11] = array2d[:, 22] / array2d[:, 4]
    # mftw_items / items
    new_array2d[:, 12] = array2d[:, 23] / array2d[:, 4]
    # wspt_items / items
    new_array2d[:, 13] = array2d[:, 24] / array2d[:, 4]
    # mspt_items / items
    new_array2d[:, 14] = array2d[:, 25] / array2d[:, 4]
    # curvy_items / items
    # Curvy item has a strong correlation with gender, however they are very right-skewed use np.power(1/6) to smooth it
    new_array2d[:, 15] = np.power(array2d[:, 26] / array2d[:, 4], 1 / 6)
    # sacc_items / items
    new_array2d[:, 16] = array2d[:, 27] / array2d[:, 4]
    # msite_orders / orders
    new_array2d[:, 17] = array2d[:, 28] / array2d[:, 3]
    # desktop_orders / orders
    new_array2d[:, 18] = array2d[:, 29] / array2d[:, 3]
    # android_orders / orders
    new_array2d[:, 19] = array2d[:, 30] / array2d[:, 3]
    # ios_orders / orders
    new_array2d[:, 20] = array2d[:, 31] / array2d[:, 3]
    # other_device_orders / orders
    new_array2d[:, 21] = array2d[:, 32] / array2d[:, 3]
    # work_orders / orders
    new_array2d[:, 22] = array2d[:, 33] / array2d[:, 3]
    # home_orders / orders
    new_array2d[:, 23] = array2d[:, 34] / array2d[:, 3]
    # parcelpoint_orders / orders
    new_array2d[:, 24] = array2d[:, 35] / array2d[:, 3]
    # other_collection_orders / orders
    new_array2d[:, 25] = array2d[:, 36] / array2d[:, 3]
    # average_discount_onoffer / orders
    new_array2d[:, 26] = array2d[:, 39]
    # average_discount_used / orders
    new_array2d[:, 27] = array2d[:, 40]
    # revenue / order
    new_array2d[:, 28] = array2d[:, 41] / array2d[:, 3]

    # normalize by each feature
    new_array2d = normalize(new_array2d, axis=0, norm='max')
    return new_array2d


def feature_selection_classifier_1(array2d):
    """
    For the first classifier we select gender related features as base
    By eigen decomposition we see a few features either has too much unintended contribution or negative contribution
    e.g. parcel shipment features being the biggest contributor or male items not contribute to male gender labeling
    The are removed
    See comment for details features selected
    :param array2d: data with all features
    :return: the selected feature
    """
    newArray2d = np.zeros([array2d.shape[0], 10])
    # male_items / female_items + male_items
    newArray2d[:, 0] = array2d[:, 5]
    # wapp_items / items
    newArray2d[:, 1] = array2d[:, 7]
    # wftw_items / items
    newArray2d[:, 2] = array2d[:, 8]
    # mapp_items / items
    newArray2d[:, 3] = array2d[:, 9]
    # wacc_items / items
    newArray2d[:, 4] = array2d[:, 10]
    # macc_items / items
    newArray2d[:, 5] = array2d[:, 11]
    # mftw_items / items
    newArray2d[:, 6] = array2d[:, 12]
    # curvy_items / items
    newArray2d[:, 7] = array2d[:, 15]
    # average_discount_onoffer / orders
    newArray2d[:, 8] = array2d[:, 26]
    # average_discount_used / orders
    newArray2d[:, 9] = array2d[:, 27]

    return newArray2d


def feature_selection_classifier_2(array2d):
    """
    For the second classifier we select gender related features plus cost effective items like discount rate, cancel/order
    By eigen decomposition we see a few features either has too much unintended contribution or negative contribution
    e.g. parcel shipment features being the biggest contributor or male items not contribute to male gender labeling
    The are removed
    See comment for details
    :param array2d: data with all features
    :return: the selected feature
    """
    newArray2d = np.zeros([array2d.shape[0], 16])
    # items/ orders
    newArray2d[:, 0] = array2d[:, 0]
    # cancels / orders
    newArray2d[:, 1] = array2d[:, 1]
    # returns / items
    newArray2d[:, 2] = array2d[:, 2]
    # voucher / orders
    newArray2d[:, 3] = array2d[:, 3]
    # female_items / female_items + male_items
    # newArray2d[:, 4] = array2d[:, 4]
    # male_items / female_items + male_items
    newArray2d[:, 4] = array2d[:, 5]
    # wapp_items / items
    newArray2d[:, 5] = array2d[:, 7]
    # wftw_items / items
    newArray2d[:, 6] = array2d[:, 8]
    # mapp_items / items
    newArray2d[:, 7] = array2d[:, 9]
    # wacc_items / items
    newArray2d[:, 8] = array2d[:, 10]
    # macc_items / items
    newArray2d[:, 9] = array2d[:, 11]
    # mftw_items / items
    newArray2d[:, 10] = array2d[:, 12]
    # mspt_items / items
    newArray2d[:, 11] = array2d[:, 14]
    # curvy_items / items
    newArray2d[:, 12] = array2d[:, 15]
    # average_discount_onoffer / orders
    newArray2d[:, 13] = array2d[:, 26]
    # average_discount_used / orders
    newArray2d[:, 14] = array2d[:, 27]
    # revenue / order
    newArray2d[:, 15] = array2d[:, 28]
    return newArray2d


def feature_selection_classifier_3(array2d):
    """
    For the last classifier we select gender related features plus payment method and shipment related features
    By eigen decomposition we see a few features either has too much unintended contribution or negative contribution
    e.g. parcel shipment features being the biggest contributor or male items not contribute to male gender labeling
    The are removed
    See comment for details
    :param array2d: data with all features
    :return: the selected feature
    """
    newArray2d = np.zeros([array2d.shape[0], 18])
    # female_items / female_items + male_items
    newArray2d[:, 0] = array2d[:, 4]
    # male_items / female_items + male_items
    newArray2d[:, 1] = array2d[:, 5]
    # wapp_items / items
    newArray2d[:, 2] = array2d[:, 7]
    # wftw_items / items
    newArray2d[:, 3] = array2d[:, 8]
    # mapp_items / items
    newArray2d[:, 4] = array2d[:, 9]
    # wacc_items / items
    newArray2d[:, 5] = array2d[:, 10]
    # macc_items / items
    newArray2d[:, 6] = array2d[:, 11]
    # mftw_items / items
    newArray2d[:, 7] = array2d[:, 12]
    # curvy_items / items
    newArray2d[:, 8] = array2d[:, 15]
    # msite_orders / orders
    newArray2d[:, 9] = array2d[:, 17]
    # desktop_orders / orders
    newArray2d[:, 10] = array2d[:, 18]
    # android_orders / orders
    newArray2d[:, 11] = array2d[:, 19]
    # ios_orders / orders
    newArray2d[:, 12] = array2d[:, 20]
    # other_device_orders / orders
    newArray2d[:, 13] = array2d[:, 21]
    # home_orders / orders
    newArray2d[:, 14] = array2d[:, 23]
    # other_collection_orders / orders
    newArray2d[:, 15] = array2d[:, 25]
    # average_discount_onoffer / orders
    newArray2d[:, 16] = array2d[:, 26]
    # average_discount_used / orders
    newArray2d[:, 17] = array2d[:, 27]
    return newArray2d


def kmeans_classification(nb_clusters, data2d):
    """
    The K means classifier
    Clustering assign random labels to female (0 or 1), the prior knowledge tells us males < half of the samples
    we use the knowledge to assign 0 to female, 1 to male
    :param nb_clusters: number of clusters(gender), 2
    :param data2d: data to be classified
    :return: labeled data
    """
    num_clusters = nb_clusters
    model = KMeans(n_clusters=num_clusters, verbose=0, random_state=6)
    model.fit(data2d)
    labels = model.labels_
    # Clustering assign random labels to female (0 or 1), the prior knowledge tells us male are half of the samples
    # we use the knowledge to assign 0 to female, 1 to male
    if sum(labels) > (data2d.shape[0]) / 2:
        labels = 1 - labels
    print("First 30 result, female = 0")
    print(*labels[:30])
    return labels


def minibatchkmeans_classifier(nb_clusters, data2d):
    """
    The Mini Batch K means classifier, we would like to try different feature and different classifier for
    Clustering assign random labels to female (0 or 1), the prior knowledge tells us males < half of the samples
    we use the knowledge to assign 0 to female, 1 to male
    :param nb_clusters: number of clusters(gender), 2
    :param data2d: data to be classified
    :return: labeled data
    """
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=nb_clusters, n_init=10, max_no_improvement=20, random_state=2)
    mbk.fit(data2d)
    labels = mbk.labels_
    # Clustering assign random labels to female (0 or 1), the prior knowledge tells us male are half of the samples
    # we use the knowledge to assign 0 to female, 1 to male
    if sum(labels) > (data2d.shape[0]) / 2:
        labels = 1 - labels
    print("First 30 results, female = 0")
    print(*labels[:30])
    return labels


def agglomerative_clustering(nu_clusters, data2d):
    """
    The agglomerative_clustering classifier, the code is provided but not used
    Clustering assign random labels to female (0 or 1), the prior knowledge tells us males < half of the samples
    we use the knowledge to assign 0 to female, 1 to male
    :param nb_clusters: number of clusters(gender), 2
    :param data2d: data to be classified
    :return: labeled data
    """
    agglomerative = AgglomerativeClustering(n_clusters=nu_clusters, linkage="ward", memory=".")
    agglomerative.fit(data2d[:])
    # Transform our data to list form and store them in results list
    labels = agglomerative.labels_
    print(sum(labels))
    if sum(labels) > (data2d.shape[0]) / 2:
        labels = 1 - labels
    return labels


# Plot result with PCA in 3D
def plot3d(data, label, feature0, feature1, feature2):
    """
    Visualize data in 3D with label female= red dot, male = blue cross
    Only first 4000 data are plotted for performance
    :param data: data to be plotted
    :param label: label associated with data
    :param feature0: feature 1 to be plotted
    :param feature1: feature 2 to be plotted
    :param feature2: feature 3 to be plotted
    :return:
    """
    female = []
    male = []
    for i in range(0, 4000):
        if label[i] == 0:
            female.append([data[i, 0], data[i, 1], data[i, 2]])
        elif label[i] == 1:
            male.append([data[i, 0], data[i, 1], data[i, 2]])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot([row[feature0] for row in female], [row[feature1] for row in female], [row[feature2] for row in female],
            'o', markersize=8, color='red',
            alpha=0.5, label='Female')
    ax.plot([row[feature0] for row in male], [row[feature1] for row in male], [row[feature2] for row in male], '+',
            markersize=8, alpha=0.5,
            color='blue', label='Male')
    plt.title('4000 Samples for Female and Male')
    ax.legend(loc='upper right')
    plt.show()


def plot2d(data, labels, feature0, feature1):
    """
    Visualize data in 2D with label female= red dot, male = blue cross
    Only first 4000 data are plotted for performance
    :param data: data to be plotted
    :param label: label associated with data
    :param feature0: feature 1 to be plotted
    :param feature1: feature 2 to be plotted
    :param feature2: feature 3 to be plotted
    :return:
    """
    for i in range(0, 4000):
        if labels[i] == 0:
            female = pl.scatter(data[i, feature0], data[i, feature1], c='r', marker='o')
        elif labels[i] == 1:
            male = pl.scatter(data[i, feature0], data[i, feature1], c='b', marker='+')
    pl.legend([female, male], ['Female', 'Male'])
    pl.title('4000 Samples for Female and Male')
    pl.show()


def eigen_decomposition(X, features):
    """
    Using eigen value decomposition to find which features contribute most to the cluster result
    A * eigenvectors = eigenvalue * eigenvectors
    The eigenvectors shows how much each feature contribute to the linear transformation
    Since eigenvalue are sorted, we actually see the weight of each feature's impact in the order of  descending variance
    Parameters
    :param X: the matrix to be eigen_decomposed
    :param X: the matrix to be eigen_decomposed
    """
    # Center to average
    Xctr = X - X.mean(0)
    # covariance matrix
    Xcov = np.cov(Xctr.T)

    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = sp.linalg.eigh(Xcov)

    # Sort the eigenvalues and the eigenvectors descending
    sortedindex = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sortedindex]
    eigen_vectors = eigen_vectors[:, sortedindex]

    ###########
    y_pos = np.arange(len(features))
    weight = eigen_vectors[0]

    figure, axis = plt.subplots(2, 1)

    axis[0].bar(features, eigen_vectors[0])
    plt.setp(axis[0], title="First and Second Component's Eigenvectors ", ylabel='Weight')
    axis[0].set_xticks(features, features)
    axis[1].bar(features, eigen_vectors[1])
    axis[1].set_xticks(features, features)
    plt.setp(axis[1], ylabel='Weight')
    # axis[0].bar(y_pos, weight, align='center', alpha=0.5)
    # axis[0].xticks(y_pos, features)
    # axis[0].ylabel('Weight')
    # axis[0].title('Features')
    #
    # axis[1].bar(y_pos, weight, align='center', alpha=0.5)
    # axis[1].xticks(y_pos, features)
    # axis[1].ylabel('Weight')
    # axis[1].title('Features')

    plt.show()
    # return eigen_values, eigen_vectors
