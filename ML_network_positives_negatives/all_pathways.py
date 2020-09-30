from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import pydotplus as pydotplus
import seaborn as sns
import pandas as pd
import numpy as np
import random
from optparse import OptionParser
import pickle,os


random.seed(0)


def colors(n):
    """
    Generate n random distinct rgb colors
    :param n: number of color to generate
    :return: list of rgb colors
    """
    ret = []
    red = int(random.random() * 256)
    green = int(random.random() * 256)
    blue = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        red += step
        green += step
        blue += step
        red = int(red) % 256
        green = int(green) % 256
        blue = int(blue) % 256
        ret.append([red, green, blue])
    return np.asarray(ret)


def histogram(value, xlabel, name=None, ylabel=None, title=None):
    """
    Print histogram with value attach to x-axis's labels
    :param value: list of value
    :param xlabel: x-axis's labels
    :param name: file's name
    :param ylabel: y-axis's label
    :param title: histogram's title
    """

    # Setting the positions and width for the bars and color
    pos = list(range(len(xlabel)))
    width = 0.25
    c = 'red'
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar in position pos + some width buffer,
    rects = plt.bar([p + width for p in pos],
                    #using value data,
                    value,
                    # of width
                    width,
                    # with alpha 0.5
                    alpha=0.5,
                    # with color
                    color=c)

    # Set the y axis label
    if ylabel is None:
        ax.set_ylabel('features importance')
    else:
        ax.set_ylabel(ylabel)

    if not title is None:
        # Set the chart's title
        ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(xlabel)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)

    ax.axhline(0, color='black')

    plt.tight_layout()

    # Adding the legend and showing the plot

    plt.grid(ls='dotted')

    if name is None:
        fig.savefig("histogram_" + title + ".png")
    else:
        fig.savefig(name + ".png")

    plt.close(fig)


def plot_correlation_matrix(corr, name):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, ax=ax,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title(name + "'s correlation")
    plt.tight_layout()
    f.savefig(os.path.join(save_dir,dme+'_'+name), dpi=400)
    plt.close()


def visualize_tree(tree, feature_names, name=None):
    """
    Create a png of a tree plot using graphviz
    :param tree: scikit-learn DecsisionTree
    :param feature_names: list of feature names
    :param name: name of file .png
    :param n_model: number of model used to predict (only for classification with one model per output)
    """

    ########################################
    ##  Color of nodes:                   ##
    ##  - Orange for negative prediction  ##
    ##  - Blue for positive prediction    ##
    ########################################

    f = export_graphviz(tree, out_file=None,
                        feature_names=feature_names,
                        filled=True,
                        rounded=True)

    graph = pydotplus.graph_from_dot_data(f)

    if name is not None:
        #graph.write_dot(name+".dot")
        graph.write_png(name+".png")
    else:
        #graph.write_dot("decisionTreePlot.dot")
        graph.write_png("decisionTreePlot.png")


def pca_plot(df_input, df_target, n_comp, name, color=None):

    if not(n_comp == 2 or n_comp == 3):
        raise ValueError("n_comp can be only 2 or 3")

    # Compute PCA
    pca = PCA(n_components=n_comp)
    principal_components = pca.fit_transform(df_input)

    principal_df = pd.DataFrame(data=principal_components,
                                columns=['pc' + str(i+1) for i in range(n_comp)])
    df_final = pd.concat([principal_df, df_target], axis=1)

    # Create plot
#    fig = plt.figure(figsize=(10, 8))
#    if n_comp == 3:
#        fig, ax = plt.subplots(figsize=(10, 8),projection='3d') # modified 11-25-19 JLW
#    else:
#        fig, ax = plt.subplots(figsize=(10, 8))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlabel('Component 1 (' + format(pca.explained_variance_ratio_[0], '02f') + ')', fontsize=15)
    ax.set_ylabel('Component 2 (' + format(pca.explained_variance_ratio_[1], '02f') + ')', fontsize=15)
#    if n_comp > 2:
#        ax.set_zlabel('Principal Component 3 (' + format(pca.explained_variance_ratio_[2], '02f') + ')', fontsize=15)

    title = str(n_comp) + ' Component PCA: Target (' + format(pca.explained_variance_ratio_.cumsum()[1], '02f') + ')'
    if 'cluster' in name.lower() or 'kmeans' in name.lower():
        title = "cluster - " + title
    ax.set_title(title, fontsize=20)

    d = {}
    unique_target = df_target.unique()
    for u in unique_target:
        d[u] = {'pc1': [], 'pc2': []}
        if n_comp > 2:
            d[u]['pc3'] = []

    f = lambda n: enumerate(zip(df_final["pc1"], df_final["pc2"], df_final["pc2"])) if n == 2 else enumerate(zip(df_final["pc1"], df_final["pc2"], df_final["pc3"]))
    for i, (pc1, pc2, pc3) in f(n_comp):
        for ut in unique_target:
            if df_target[i] == ut:
                d[ut]['pc1'].append(pc1)
                d[ut]['pc2'].append(pc2)
                if n_comp > 2:
                    d[ut]['pc3'].append(pc3)
                continue

    if color is None:
        color = colors(len(unique_target))

    label_fun = lambda j, name: 'Cluster ' + str(j) if 'cluster' in name.lower() or 'kmeans' in name.lower() else 'Positive' if j == 1 else 'Negative'
    for i, ut in enumerate(unique_target):
        if n_comp == 3:
            ax.scatter(d[ut]['pc1'], d[ut]['pc2'], d[ut]['pc3'], c=[color[i] / 255.0] * len(d[ut]['pc2']), label=label_fun(ut, name))
        else:
            ax.scatter(d[ut]['pc1'], d[ut]['pc2'], c=[color[i]/255.0] * len(d[ut]['pc2']), label=label_fun(ut, name))

    plt.legend(loc='best')
    savefig_path = os.path.join(save_dir,dme+'_'+name+"_3D.png") if n_comp == 3 else os.path.join(save_dir,dme+'_'+name+".png")
    plt.savefig(savefig_path, dpi=400)
    plt.close(fig)


def run_model(model, x_tr, y_tr, x_ts, y_ts, fcol):
    model_name = str(type(model)).split(".")[-1].replace("'>", "")
    print("\n", model_name, "/t/t", model.get_params())
    model.fit(x_tr, y_tr)
#    reg_coef = model.coef_ # save regression coefficients
    reg_coef = list(zip(model.coef_[0], list(fcol)))
    pickle.dump(reg_coef,open(os.path.join(save_dir,dme+'_feat_imp_scores_092920.pkl'),'wb'))

    prediction_tr = model.predict(x_tr)

    if x_ts is not None and y_ts is not None:
        prediction_ts = model.predict(x_ts)
    else:
        y_ts = None
        prediction_ts = None

    # get y_scores for roc_curve, this only works for logistic regression
    y_score = model.fit(x_tr, y_tr).decision_function(x_ts) 
    y_score_data = zip(y_ts,y_score)
    pickle.dump(y_score_data,open(os.path.join(save_dir,dme+'_ylabels_scores_072720.pkl'),'wb'))
    [fpr, tpr, thresholds] = roc_curve(y_ts, y_score)
    roc_data = zip(fpr,tpr)
    pickle.dump(roc_data,open(os.path.join(save_dir,dme+'_roc_data_072720.pkl'),'wb'))

    scoring(y_tr, prediction_tr, y_ts, prediction_ts)
    if "Logistic" in model_name:
        fimp = model.coef_[0]
        #print(str(list(zip(fcol, fimp))))
        pickle.dump(zip(fcol,fimp),open(os.path.join(save_dir,dme+'_feature_import.pkl'),'wb'))
    else:
        fimp = model.feature_importances_
        # print(str(list(zip(fcol, fimp))))
    histogram(fimp, fcol, title="features_"+model_name)

    if "Decision" in model_name:
        visualize_tree(model, fcol)


def scoring(y_tr, prediction_tr, y_ts, prediction_ts):
    for y_true, res, phase in [(y_tr, prediction_tr, "TR"), (y_ts, prediction_ts, "TS")]:
        if y_ts is None or prediction_ts is None:
            continue
        print(phase)
        cm = confusion_matrix(y_true, res)
        # printerTree.plot_confusion_matrix(cm,[0,1])
        print(" confusion matrix: \n", cm)
        print(" accuracy: ", accuracy_score(y_true, res))
        print(" f1_score: ", f1_score(y_true, res, average="micro"))
        print(" roc_auc_score: ", roc_auc_score(y_true, res))

    # save results for later plotting
    tr_acc_score = accuracy_score(y_tr,prediction_tr)
    tr_f1_score = f1_score(y_tr,prediction_tr,average="micro")
    tr_roc_auc_score = roc_auc_score(y_tr,prediction_tr)

    ts_acc_score = accuracy_score(y_ts, prediction_ts)
    ts_f1_score = f1_score(y_ts, prediction_ts,average="micro")
    ts_roc_auc_score = roc_auc_score(y_ts, prediction_ts)

    pickle.dump(tr_acc_score,open(os.path.join(save_dir,dme+'_tr_acc_score.pkl'),'wb'))
    pickle.dump(tr_f1_score,open(os.path.join(save_dir,dme+'_tr_f1_score.pkl'),'wb'))
    pickle.dump(tr_roc_auc_score,open(os.path.join(save_dir,dme+'_tr_roc_auc_score.pkl'),'wb'))
    pickle.dump(ts_acc_score,open(os.path.join(save_dir,dme+'_ts_acc_score.pkl'),'wb'))
    pickle.dump(ts_f1_score,open(os.path.join(save_dir,dme+'_ts_f1_score.pkl'),'wb'))
    pickle.dump(ts_roc_auc_score,open(os.path.join(save_dir,dme+'_ts_roc_auc_score.pkl'),'wb'))

def main():
    parser = OptionParser(usage="usage: %prog [options] filename",
        version="%prog 1.0")
    parser.add_option("-m", "--mtype",
        action="store",
        dest="model_type",
        default='log_reg',
        help="model type")
    parser.add_option("-n","--dname",
        action="store",
        dest="dme_name",
        default=None,
        type = 'str',
        help="the name of the dme for saving results")
    (options, args) = parser.parse_args()
    # parse file name from input arguments
    model_type = options.model_type
    path = args[0]
    global save_dir
    save_dir = os.path.join('.',model_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Read dataset
#    path = "./ML_network_positives_negatives/Tardive_dyskinesia.txt"
    df = pd.read_csv(path, sep="\t")
    global dme
    dme = options.dme_name
    print(dme)

    # Identify features columns and target columns
    fcol = df.columns[2:]
    tcol = 'label'

    # Convert target into numeric value
    df.loc[df['label'] == 'positive', 'label'] = 1
    df.loc[df['label'] == 'negative', 'label'] = 0

    # Plot correlation
    plot_correlation_matrix(df[1:].corr(method="pearson"), "Pearson")
    plot_correlation_matrix(df[1:].corr(method="spearman"), "Spearman")

    # Split into test and training set
    x_tr, x_ts, y_tr, y_ts = train_test_split(df[fcol], df[tcol], test_size=0.2, stratify=df[tcol], random_state=0)
    y_tr=y_tr.astype('int') # modified JLW
    y_ts=y_ts.astype('int') # modified JLW

    print('Num for training, testing')
    print(len(x_tr), len(x_ts))
###    exit()

    if model_type == 'log_reg':
        # Logistic Regression model
        print('Logistic regression')
        run_model(LogisticRegression(random_state=0), x_tr, y_tr, x_ts, y_ts, fcol)

    elif model_type == 'dec_tree':
        # Decision Tree model
        print('Decision tree')
        run_model(DecisionTreeClassifier(random_state=0), x_tr, y_tr, x_ts, y_ts, fcol)

    elif model_type == 'rand_for':
        # Random Forest model
        print('Random forest')
        run_model(RandomForestClassifier(random_state=0), x_tr, y_tr, x_ts, y_ts, fcol)


if __name__ == '__main__':
    main()

## original code from Alessio
#if __name__ == '__main__':
#
#    # Read dataset
#    path = "./ML_network_positives_negatives/Tardive_dyskinesia.txt"
#    df = pd.read_csv(path, sep="\t")
#
#    # Identify features columns and target columns
#    fcol = df.columns[2:]
#    tcol = 'label'
#
#    # Convert target into numeric value
#    df.loc[df['label'] == 'positive', 'label'] = 1
#    df.loc[df['label'] == 'negative', 'label'] = 0
#
#    # Plot correlation
#    plot_correlation_matrix(df[1:].corr(method="pearson"), "Pearson")
#    plot_correlation_matrix(df[1:].corr(method="spearman"), "Spearman")
#
#    # Print PCA scatter plot
#    color = colors(2)
#    pca_plot(df[fcol], df[tcol], 2, "dyskinesia", color)
#    pca_plot(df[fcol], df[tcol], 3, "dyskinesia", color)
#
#    # Split into test and training set
#    x_tr, x_ts, y_tr, y_ts = train_test_split(df[fcol], df[tcol], test_size=0.2, stratify=df[tcol], random_state=0)
#    y_tr=y_tr.astype('int') # modified JLW
#    y_ts=y_ts.astype('int') # modified JLW
#
#    print('Num for training, testing')
#    print(len(x_tr), len(x_ts))
####    exit()
#
#    # Logistic Regression model
#    print('Logistic regression')
#    run_model(LogisticRegression(random_state=0), x_tr, y_tr, x_ts, y_ts, fcol)
#
#    # Decision Tree model
#    print('Decision tree')
#    run_model(DecisionTreeClassifier(random_state=0), x_tr, y_tr, x_ts, y_ts, fcol)
#
#    # Random Forest model
#    print('Random forest')
#    run_model(RandomForestClassifier(random_state=0), x_tr, y_tr, x_ts, y_ts, fcol)
#
#    # Clustering
#    n_clust = 3
#    km = KMeans(n_clusters=n_clust, init='random',
#                n_init=10, max_iter=300,
#                tol=1e-04, random_state=0)
#
#    y_km = km.fit_predict(df[fcol])
#
#    df['Kmeans'] = y_km
#
#    # Print PCA scatter plot over the clustering
#    color = colors(n_clust)
#    pca_plot(df[fcol], df['Kmeans'], 2, "dyskinesia_kmeans", color)
#    pca_plot(df[fcol], df['Kmeans'], 3, "dyskinesia_kmeans", color)
#
#
#
#
