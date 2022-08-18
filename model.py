import numpy as np
import xgboost as xgb
import pickle
import gzip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# load tf-idf model
with gzip.open('./models/tfidf.pgz', 'r') as f:
    tf_idf_vect = pickle.load(f)

def trans(df):
    final_tf_idf = tf_idf_vect.transform(df[0])
    dtest = xgb.DMatrix(final_tf_idf)
    return dtest

######################
# load xgb model
with gzip.open('./models/xgb-invoice.pgz', 'r') as f:
    xgboostModel = pickle.load(f)

def predict(dtest):
    pred = xgboostModel.predict(dtest)
    # load label encoder
    encoder = LabelEncoder()
    encoder.classes_ = np.load('./models/classes.npy', allow_pickle=True)
    out = encoder.inverse_transform(np.argmax(pred, axis=1))[0]
    #encoder.inverse_transform(np.array(pred))[0][0]
    return out

###############
