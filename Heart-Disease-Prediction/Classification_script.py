import numpy as np
import matplotlib.pyplot as plt
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics import auc,roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


def Confusion_matrix(y_test,pred,p_label,n_label):
    tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
    con_matrix=pd.DataFrame({'Pred_'+p_label:[tp,fp],'Pred_'+n_label:[fn,tn]},index=[p_label,n_label])
    return con_matrix

def performance_metrics(y_test,pred):
    #return acc,pre,recall,f1
    acc=accuracy_score(y_test,pred)
    pre=precision_score(y_test,pred)
    recall=recall_score(y_test,pred)
    f1=f1_score(y_test,pred)
    return acc,pre,recall,f1

def plot_roc_curve(model,X_test,y_test):
    
    pred_prob=model.predict_proba(X_test)[:,1]
    auc=roc_auc_score(y_test,pred_prob)
    fpr,tpr,thresholds=roc_curve(y_test,pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label='AUC = %0.3f' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')
    
def drawTree(model,X,path):
    os.environ["Path"] += os.pathsep + path
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=list(X.columns))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())

def saveTree(imageName):
    graph.write_png(imageName)

def featureImportance(X,tree_model):
    feature_importance=list(zip(X.columns,tree_model.feature_importances_))
    feature_importance= pd.DataFrame(feature_importance,columns=['Feature Name','Importance'])
    feature_importance=feature_importance.sort_values('Importance',ascending=False)
    plt.figure(figsize=(12,6))
    sns.barplot(x='Feature Name',y='Importance',data=feature_importance)
    plt.xlabel('Feature Name')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.title("Decision Classifier - Features Importance")

    
    