# Invoice product category classfication with Flask Python

We use the Kaggle dataset https://www.kaggle.com/datasets/nikhil1011/predict-product-category-from-given-invoice to predict product category from given item descriptions of invoice.

We also integrate with Flask to reuse the model. Note that there are three files in the models folder: 
- tf-idf.pgz for term frequency and inverse document frequency
- xgb-invoice.pgz for classification
- classes.npy for converting the encoded category back to original category

Below is the Screenshot we tested on Postman.
![image](https://user-images.githubusercontent.com/91658005/185371209-60dd6708-bc95-444a-b1c3-19cf51d75a5f.png)

