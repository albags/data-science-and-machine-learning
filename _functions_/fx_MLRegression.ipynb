{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn import linear_model\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import RidgeCV\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.linear_model import ElasticNet\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def metrics_train(X_train,y_train, model):\n",
    "    \n",
    "    # Let's calculate the metrics with our TRAIN dataset\n",
    "    y_predTrain = model.predict(X_train)   \n",
    "    cv_scores = cross_val_score(model, X_train, \n",
    "                            y_train,cv=10, scoring='r2') # Let's define the K and the \n",
    "\n",
    "    cv_scores= round(np.mean(cv_scores),3)\n",
    "    \n",
    "    statist_train = []\n",
    "    MAE_lTrain = metrics.mean_absolute_error(y_train, y_predTrain)\n",
    "    MSE_lTrain = metrics.mean_squared_error(y_train,y_predTrain)\n",
    "    RMSE_lTrain = np.sqrt(metrics.mean_squared_error(y_train, y_predTrain))\n",
    "    R2_lTrain = model.score(X_train, y_train)\n",
    "    \n",
    "    list_metrics = [MAE_lTrain, MSE_lTrain, RMSE_lTrain,\n",
    "                    R2_lTrain,cv_scores]\n",
    "    statist_train.append(list_metrics)\n",
    "    statist_train = pd.DataFrame(statist_train,\n",
    "                                 columns = ['MAE', 'MSE', \n",
    "                                            'RMSE', 'R2', 'CV_R2'],\n",
    "                                 index = ['Train'])\n",
    "    \n",
    "    return statist_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def metrics_test(X_test,y_test, model):\n",
    "    # Let's calculate the metrics with our TRAIN dataset\n",
    "    y_pred = model.predict(X_test)\n",
    "    cv_scores = cross_val_score(model, X_test, \n",
    "                            y_test,cv=10, scoring='r2') # Let's define the K and the \n",
    "\n",
    "    cv_scores= round(np.mean(cv_scores),3)\n",
    "    \n",
    "    statist_test = []\n",
    "    MAE = metrics.mean_absolute_error(y_test, y_pred)\n",
    "    MSE = metrics.mean_squared_error(y_test, y_pred)\n",
    "    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))\n",
    "    R2 = model.score(X_test,y_test)\n",
    "    \n",
    "    list_metrics = [MAE, MSE, RMSE, R2, cv_scores]\n",
    "    statist_test.append(list_metrics)\n",
    "    statist_test = pd.DataFrame(statist_test,columns = ['MAE', 'MSE', 'RMSE', 'R2', 'CV_R2'], index = ['Test'])\n",
    "    \n",
    "    return statist_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Allmetrics(model,X_train,y_train,X_test,y_test):\n",
    "    \n",
    "    stats_train = metrics_train(X_train,y_train,model)\n",
    "    stats_test = metrics_test(X_test,y_test,model)\n",
    "    final_metrics = pd.concat([stats_train,stats_test])\n",
    "    return final_metrics"
   ]
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
