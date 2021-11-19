# -*- coding: utf-8 -*-
"""
Wrapper Class to Cluster Data and to Generate Output
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2021-11-19 13:16:36 RLafarguette"
"""
###############################################################################
#%% Modules
###############################################################################
# Base
import pandas as pd
import numpy as np
import random
import string

# Functional imports
from dataclasses import dataclass, field
from typing import Any, List
from collections import Counter, OrderedDict

# Clustering-related functions
from scipy.spatial.distance import cdist, pdist

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from sklearn.neighbors import NearestCentroid

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples, silhouette_score

# Graphics and options
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set(style="white", font_scale=2)

###############################################################################
#%% Parameters
###############################################################################
# I have problems with resetting iterators (cyclers) for consistent style
# Hence I just create long lists...
line_l = ['solid', 'dotted', 'dashed', 'dashdot', 'densely dashdotted',
          'loosely dashdotdotted']*15

color_l = ['tab:blue', 'tab:red', 'tab:green', 'tab:brown', 'tab:purple',
           'tab:orange', 'tab:pink']*15

marker_l = ['o', 'v', 's', 'D', 'p', 'P', 'h', '*']*15

###############################################################################
#%% Ancillary functions
###############################################################################
def scale_wd(df, mean_d, std_d):
    """ Standard Scaler with Dictionaries (to rescale easily) """
    # Check if the inputs are correct
    mv0 = 'Dictionaries of mean and std have different keys'
    assert sorted(mean_d.keys())==sorted(std_d.keys()), mv0


    mv1 = 'Dictionaries of variables are different from columns'
    assert sorted(mean_d.keys())==sorted(df.columns), mv1

    # Standard scaling (remove mean, divide by standard deviation)
    dscale = pd.DataFrame(index=df.index, columns=df.columns)
    for var in df.columns:
        dscale[var] = (df[var] - mean_d[var])/std_d[var]
    return(dscale)

def unscale_wd(df, mean_d, std_d):
    """ Unscale with Dictionaries (to rescale easily) """
    # Check if the inputs are correct
    mv0 = 'Dictionaries of mean and std have different keys'
    assert sorted(mean_d.keys())==sorted(std_d.keys()), mv0

    mv1 = 'Dictionaries of variables are different from columns'
    assert sorted(mean_d.keys())==sorted(df.columns), mv1

    # Standard rescaling (multiply by standard deviation and add mean)
    duscale = pd.DataFrame(index=df.index, columns=df.columns)
    for var in df.columns:
        duscale[var] = (df[var]*std_d[var]) + mean_d[var]
    return(duscale)

def lab_reo(labels_l):    
    """ Reorder labels by frequency and add 1"""    
    cd = Counter(labels_l) # Frequency
    sc = OrderedDict(sorted(cd.items(), key=lambda kv: kv[1], reverse=True))

    change_d = {old:new for new, old in enumerate(sc.keys())}
    new_labels_l = [change_d[lab] + 1 for lab in labels_l]

    return(new_labels_l)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    return(None)

def random_text(length=6):
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return(x)

###############################################################################
# Cluster: Master class
###############################################################################
@dataclass
class Cluster: # Define the master cluster class
    features_l:List[str]
    df: pd.DataFrame
    random_state: int=42 # Answer to the Ultimate Question of Life :-)
    
    __description = "Cluster Analysis Wrapper with Performance Metrics"
    __author = "Romain Lafarguette, https://romainlafarguette.github.io/"
    
    def __post_init__(self): # Post-initialize the attributes
        self.dfg = self.df[self.features_l].dropna()
        self.mean_d = {v: np.mean(self.df[v]) for v in self.features_l}
        self.std_d = {v: np.std(self.df[v]) for v in self.features_l}
        self.x = scale_wd(df=self.dfg,
                          mean_d=self.mean_d, std_d=self.std_d).values
        self.pca_fit = PCA(n_components=2).fit(self.x) # PCA Fit
        self.dpca = pd.DataFrame(scale(self.pca_fit.transform(self.x)),
                                 columns=['first_comp', 'second_comp'],
                                 index=self.dfg.index)

        # Unit tests
        self.__cluster_unit_test__() # UnitTests defined below
        
    # Class variable defined as methods
    def fit(self, method='kmeans', n_clusters=2): # Class variable
        return(ClusterFit(__cluster__=self, method=method,
                          n_clusters=n_clusters))

    # Multifit class (fit with multiple clusters)
    def multifit(self, method='kmeans', clusters_l=[2, 3, 4, 5, 6, 7]): 
        return(ClusterMultiFit(__master__=self,
                               method=method, clusters_l=clusters_l))

    # Standard methods
    def dendogram(self, axvline=None, ax=None):
        """ Plot the dendogram of the fit """
        mod = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        mod_fit = mod.fit(self.x)
        
        ax = ax or plt.gca()
        plot_dendrogram(mod_fit, truncate_mode='level',
                        orientation='right', ax=ax)
        if axvline:
            ax.axvline(x=axvline, color='tab:red', ls='--')
            
        labels_l = [list(self.dfg.index)[int(idx.get_text())]
                    for idx in ax.get_yticklabels()]
        ax.set_yticklabels(labels_l)
        ax.set_xlabel('Ward metric', fontsize='small', labelpad=20)
        ax.set_ylabel('Individuals ID', fontsize='small', labelpad=20)
        ax.set_title('Hierarchical Clustering Dendogram', y=1.02)

        return(ax)

    def __cluster_unit_test__(self): # Series of tests on a class
        assert len(self.features_l)>0, 'Regressors list should not be empty'

        mv = len(self.df[self.features_l]) - len(self.x)
        if mv>0: print(f'Missing values: {mv:b}')

        mv2 = 'Obs should be > than features'
        assert len(self.x) > len(self.features_l), mv2

        # print(f'{len(self.x):d} observations,'
        #       f'{len(self.features_l):d} features')
    
###############################################################################
# Cluster -> Single Fit class
###############################################################################
@dataclass
class ClusterFit: 
    __cluster__ : Cluster # Should be the parent class
    method : str='kmeans' # Algorithm
    n_clusters: int=2 # Number of clusters

    """ Estimate the cluster model parameters """    
    def __post_init__(self): # Post-initialize the attributes
        self.__dict__.update(self.__cluster__.__dict__) # Import attributes
        self.cluster_fit, self.centroids = self.__fit()
        self.cluster_fit.labels_ = lab_reo(self.cluster_fit.labels_) # Reorder 
        self.dfr, self.dfr_x = self.__res_frame__()
        self.labels = self.dfr_x['cluster_label'].copy().values
        self.dcentroids, self.dcentroids_unscaled = self.__dcentroids_f__() 
        
        # Some useful features saved in a dictionary
        self.color_d = self.__color_d_generator__()
        
        # Class variables: defined as attributes (not as methods)
        self.perf = ClusterFitPerformance(__clusterfitp__=self)
        self.plot = ClusterFitPlot(__clusterfit__=self)
            
    # Hidden methods to generate specific attributes in post_init
    def __fit(self): # Fit the clustering method   
        if self.method=='ward':
            ward = AgglomerativeClustering(distance_threshold=None,
                                           n_clusters=self.n_clusters,
                                           linkage='ward')
            cfit = ward.fit(self.x)            
            y = cfit.fit_predict(self.x) # Samples
            
            # Compute the centroids separately
            centroids = NearestCentroid().fit(self.x, y).centroids_
            
        elif self.method=='kmeans':
            kmeans = KMeans(init="k-means++", n_clusters=self.n_clusters,
                            n_init=4, random_state=self.random_state)
            cfit = kmeans.fit(self.x)            
            centroids = kmeans.cluster_centers_ # provided by default
            
        else:
            raise ValueError('Clustering method not recognized')
        
        return((cfit, centroids))
        
    def __res_frame__(self): # Package the results in the frame
        # Get the labels
        labels = np.reshape(self.cluster_fit.labels_, self.x.shape[0])        
        
        # Create a new dataframe with the labels : unscaled
        dfr = self.dfg.copy() # Note that dfg is unscaled (not as self.x)
        dfr.insert(0, 'cluster_label', labels) # Add the column first
        
        # Create a new dataframe with the labels : scaled
        dfr_x = pd.DataFrame(self.x)        
        dfr_x.columns = self.features_l
        dfr_x.insert(0, 'cluster_label', labels) # Add the column first    
        return((dfr, dfr_x))
       
    def __features_labels__(self, stat='mean'): # Summarize features by label
        # Summarize the features by label
        if stat=='mean':    
            dsum = self.dfr.groupby('cluster_label')[
                self.features_l].mean().copy()
        elif stat=='median':
            dsum = self.dfr.groupby('cluster_label')[
                self.features_l].median().copy()
        else:
            raise ValueError('stat should be in (mean, median)')
        return(dsum)

    def __color_d_generator__(self):
        color_d = dict()
        for i in self.cluster_fit.labels_:            
            color_d[i] = color_l[i-1] # Because the labels start at 1
        return(color_d)

    def __dcentroids_f__(self):
        """ 
        Technically the centroids could be computed as simple average
        But in case I want to change the metric, I offer more choice
        I also recompute them on uncentered variables via unscaling
        """
                
        # Scaled        
        centroids = NearestCentroid().fit(self.x, self.labels).centroids_
        dc = pd.DataFrame(centroids, columns=self.features_l).copy()
        
        # Unscaled centroids
        du = unscale_wd(df=dc, mean_d=self.mean_d, std_d=self.std_d)

        unq_labels = sorted(set(self.labels))
        dc.insert(0, 'cluster_label', unq_labels) 
        du.insert(0, 'cluster_label', unq_labels) 
        
        # In case I want to make sure the centroids are well positionned
        # But by default is ok, and time consuming...
        # # Identify the centroids with the minimum distance within a group
        # for idx in dc.index: # For each centroid
        #     dist_d = dict()
        #     for lab in sorted(set(self.dfr_x.cluster_label)): # each label
        #         cent = dc.loc[[idx], self.features_l].copy().values
        #         cond = (self.dfr_x.cluster_label==lab) # Given cluster
        #         group = self.dfr_x.loc[cond, self.features_l].copy().values
        #         dist = cdist(group, cent, 'euclidean') # Dist to centroids
        #         dist_d[lab] = np.median(dist) # distance within each group

        #     o_lab = int(min(dist_d, key=dist_d.get)) # Minimum distance   
        #     dc.iloc[idx, dc.columns.get_loc('cluster_label')] = o_lab    
        
        return((dc, du))
    
###############################################################################
# Cluster Fit -> Cluster Fit Performance 
###############################################################################
@dataclass
class ClusterFitPerformance:
    __clusterfitp__: ClusterFit # Parent class
    
    def __post_init__(self): # Initialize the attributes
        self.__dict__.update(self.__clusterfitp__.__dict__) # Import all attr
        self.db_score = davies_bouldin_score(self.x, self.labels)
        self.ch_score = calinski_harabasz_score(self.x, self.labels)        
        self.within_ss, self.between_ss, self.tot_ss = self.__sse__()
        self.silhouette_avg = silhouette_score(self.x, self.labels)
        
        # Class variable
        self.plot = ClusterFitPerformancePlot(__clusterfitperf__=self)

    def __sse__(self):
        """ Compute the Sum of Squares Metrics for Different Algorithms """
        D = cdist(self.x, self.centroids, 'euclidean') # Distance to centroids
        dist = np.min(D, axis=1) # minimum = distance to the closest centroid
        avg_within_ss = sum(dist)/self.x.shape[0]

        within_ss = sum(dist**2) # Within TSS ("_inertia" for Kmeans)
        tot_ss = sum(pdist(self.x)**2)/self.x.shape[0] # Total sum of squares
        between_ss = tot_ss - within_ss    # The between-cluster sum of squares

        return((within_ss, between_ss, tot_ss))

        
    def __silhouette_frame__(self):
        """ Compute the silhouette values and store in a frame """
        dsilh = self.dfr_x.copy()
        
        # Compute the average silhouette scores 
        _silhouette_avg = silhouette_score(self.x, self.labels)
        _silhouette_values = silhouette_samples(self.x, self.labels)

        # Update the performance frame        
        dsilh['silhouette_val'] = _silhouette_values
        dsilh['silhouette_avg'] = _silhouette_avg
        
        return(dsilh) # Just update the performance frame


###############################################################################
# Cluster -> Multi Fit class
###############################################################################
@dataclass
class ClusterMultiFit: 
    __master__: Cluster # Parent class
    method: str='kmeans' # Algorithm
    clusters_l: List[int]=field(default_factory=lambda:[2, 3, 4, 5, 6, 7])
    
    """ Estimate the cluster model parameters """    
    def __post_init__(self): # Post-initialize the attributes
        self.features_l = self.__master__.features_l
        self.df = self.__master__.df
        self.multifit_d, self.multiperf_d = self.__multifit_f__()

        # Class variable
        self.plot = ClusterMultiFitPlot(__multifit__=self)
                    
    def __multifit_f__(self):
        """ Call Cluster class independently to avoid namespace conflicts"""
        multifit_d = dict(); multiperf_d = dict()
        for n in self.clusters_l:
            cfit = Cluster(features_l=self.features_l,
                           df=self.df).fit(method=self.method,
                                           n_clusters=n)
            multifit_d[n] = cfit 
            multiperf_d[n] = cfit.perf
        return((multifit_d, multiperf_d))

###############################################################################
# Cluster Multi Fit -> Cluster Multi Fit Plot 
###############################################################################
@dataclass
class ClusterMultiFitPlot: 
    __multifit__: ClusterMultiFit # Parent class

    def __post_init__(self): # Initialize the attributes
        self.__dict__.update(self.__multifit__.__dict__) # Import all attr


    def elbow_plot(self, ax=None):
        x_l = sorted(self.multiperf_d.keys())
        y_l = [self.multiperf_d[n].within_ss for n in x_l]
        
        ax = ax or plt.gca()
        ax.plot(x_l, y_l, color='tab:blue', lw=3, ls='-', 
                marker='D', markerfacecolor='tab:red')

        ax.set_xticks(x_l)
        ax.set_xlabel('Number of clusters', labelpad=20)
        ax.set_ylabel('Inertia', labelpad=20)
        ax.set_title('Elbow Plot', y=1.02)
        return(ax)

    def silhouette_avg_plot(self, ax=None):
        x_l = sorted(self.multiperf_d.keys())
        y_l = [self.multiperf_d[n].silhouette_avg for n in x_l]

        # Initialize the plot
        ax = ax or plt.gca()
        ax.plot(x_l, y_l, color='tab:blue', lw=3, ls='-', 
                marker='D', markerfacecolor='tab:red')

        ax.set_xticks(x_l)
        ax.set_xlabel('Number of clusters', labelpad=20)
        ax.set_ylabel('Silhouette Average', labelpad=20)
        ax.set_title('Silhouette Average (the closer to 1 the better)', y=1.02)
        return(ax)
    
    def ch_plot(self, ax=None):
        x_l = sorted(self.multiperf_d.keys())
        y_l = [self.multiperf_d[n].ch_score for n in x_l]

        ax = ax or plt.gca()
        ax.plot(x_l, y_l, color='tab:blue', lw=3, ls='-', 
                marker='D', markerfacecolor='tab:red')

        ax.set_xticks(x_l)
        ax.set_xlabel('Number of clusters', labelpad=20)
        ax.set_ylabel('Calinski Harabasz Score', labelpad=20)
        ax.set_title('Calinski Harabasz Score (the higher the better)', y=1.02)
        return(ax)

    def db_plot(self, ax=None):
        x_l = sorted(self.multiperf_d.keys())
        y_l = [self.multiperf_d[n].db_score for n in x_l]

        ax = ax or plt.gca()
        ax.plot(x_l, y_l, color='tab:blue', lw=3, ls='-', 
                marker='D', markerfacecolor='tab:red')

        ax.set_xticks(x_l)
        ax.set_xlabel('Number of clusters', labelpad=20)
        ax.set_ylabel('Davies Bouldin Score', labelpad=20)
        ax.set_title('Davies Bouldin Score (the lower the better)', y=1.02)
        return(ax)

    def multifit_summary_perf_plot(self, axvline=None):
        fig, axs = plt.subplots(2, 2, sharex=True)
        ax1, ax2, ax3, ax4 = axs.ravel()

        ax1 = self.elbow_plot(ax=ax1)
        ax1.set_xlabel('')
        
        ax2 = self.silhouette_avg_plot(ax=ax2)
        ax2.set_xlabel('')
        
        ax3 = self.ch_plot(ax=ax3)

        ax4 = self.db_plot(ax=ax4)

        if axvline:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=axvline, ls='--', lw=2, color='tab:red')
        
        plt.suptitle('Summary Performance Plot for MultiFit Clusters')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        return(fig)
        
###############################################################################
# Cluster Fit -> Cluster Fit Plots 
###############################################################################
@dataclass
class ClusterFitPlot: 
    __clusterfit__: ClusterFit # Parent class
    
    def __post_init__(self): # Initialize the attributes
        self.__dict__.update(self.__clusterfit__.__dict__) # Import all attr
        
    def cluster_size(self, ax=None, **kwargs):
        var = self.features_l[0] # take the first variable
        df = self.dfr.copy()
        dcount = df.groupby('cluster_label', as_index=False)[var].count()
        dcount = dcount.rename(columns={var:'count'})
        dcount['color'] = [self.color_d[x] for x in dcount['cluster_label']]
    
        ax = ax or plt.gca() # If None returns the other choice   
        ax.bar(dcount['cluster_label'], dcount['count'], color=dcount['color'])
        ax.set_xlabel('Cluster ID', labelpad=20, fontsize='small')
        ax.set_ylabel('Number of individuals', labelpad=20, fontsize='small')
        ax.set_title('Number of individuals by cluster', y=1.02)
        ax.set_xticks(dcount.index)
        
        return(ax) # Return just the ax

    def single_feature_by_cluster(self, feature=None,
                                  label=None, stat='mean',
                                  ax=None, **kwargs):

        # Because the default value is contained in self, need to manage here
        feature = feature or self.features_l[0]
        label = label or self.features_l[0]
        
        # Retrieve the summary statistics (very fast)
        dsum = self.__clusterfit__.__features_labels__(stat=stat)
        dsum['color'] = [self.color_d[x] for x in dsum.index]
                        
        # Prepare the chart
        ax = ax or plt.gca() # If None returns the other choice
        
        ax.bar(dsum.index, dsum[feature], label=label, color=dsum['color'])
        ax.axhline(y=0, color='black')

        ax.legend(frameon=False, handlelength=1, fontsize='small')
        
        ax.set_xticks(dsum.index)
        ax.tick_params(axis='both', labelsize='small')
                
        return(ax) # Return the whole figure
    
    def all_features_by_cluster(self, stat='mean', ncols=3):
        # Retrieve the summary statistics
        dsum = self.__clusterfit__.__features_labels__(stat=stat)
        dsum['color'] = [self.color_d[x] for x in dsum.index]
        
        # Determine the chart
        nplots = len(self.features_l)
        
        # compute the number of rows required
        # nrows = nplots // ncols 
        # nrows += nplots % ncols
        if nplots % ncols == 0:
            nrows = nplots // ncols
        else:
            nrows = nplots // ncols+1
        
        fig, axs = plt.subplots(nrows, ncols, sharex=True)
        ax_l = axs.ravel()

        for idx, var in enumerate(self.features_l):
            ax = ax_l[idx]
            ax.bar(dsum.index, dsum[var], color=dsum['color'])
            ax.axhline(y=0, color='black')
            ax.set_title(var, fontsize='small')
            ax.set_xticks(dsum.index)
            ax.tick_params(axis='both', labelsize='x-small')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)    
        return(axs) # Return the whole figure


    def all_features_by_centroid(self, ncols=3):
        # Retrieve the unscaled centroid frame
        dcent = self.dcentroids_unscaled.copy()
        dcent['color'] = [self.color_d[x] for x in dcent.cluster_label]
        
        # Determine the chart
        nplots = len(self.features_l)
        
        # compute the number of rows required
        # nrows = nplots // ncols 
        # nrows += nplots % ncols
        if nplots % ncols == 0:
            nrows = nplots // ncols
        else:
            nrows = nplots // ncols+1
        
        fig, axs = plt.subplots(nrows, ncols, sharex=True)
        ax_l = axs.ravel()

        for idx, var in enumerate(self.features_l):
            ax = ax_l[idx]
            ax.bar(dcent.cluster_label, dcent[var], color=dcent['color'])
            ax.axhline(y=0, color='black')
            ax.set_title(var, fontsize='small')
            ax.set_xticks(dcent.cluster_label)
            ax.tick_params(axis='both', labelsize='x-small')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)    
        return(axs) # Return the whole figure

    
    def cluster_2d(self, style='centroids', axis_label=True,
                   ax=None, **kwargs):
        """ Present the clustering in a 2D space obtained through PCA """
        
        # Create the meshgrid
        dp = self.dpca.copy()
        
        # Obtain labels for PCA and each point 
        pca_labels_org = self.cluster_fit.fit_predict(dp.values)
        dp['labels'] = lab_reo(pca_labels_org) # Reorder them    

        # Compute the centroids as averaged 
        vi_l = ['first_comp', 'second_comp']
        x_pca = dp[vi_l].values
        y_pca = dp['labels'].values
        u_labels = sorted(set(dp['labels']))
        pca_cent = NearestCentroid().fit(x_pca, y_pca).centroids_
        dpcc = pd.DataFrame(pca_cent, index=u_labels, columns=vi_l)
        dpcc.insert(0, 'labels', dpcc.index)
        
        # Generate the plot
        ax = ax or plt.subplot(1, 1, 1) # If None returns the other choice
        
        for idx, lab in enumerate(sorted(dp['labels'].unique())):
            dpl = dp.loc[dp.labels==lab, :].copy()
            dcl = dpcc.loc[dpcc.labels==lab, :].copy()
            fcolor = self.color_d[lab]
            fmarker = marker_l[idx]
            if style=='centroids':
                # Points
                ax.scatter(dpl['first_comp'], dpl['second_comp'], s=0)
                for x, y, txt in zip(dpl['first_comp'], dpl['second_comp'],
                                     dpl.index):
                    ax.text(x, y, txt, fontsize='x-small', color=fcolor)
                
                # ax.scatter(dpl['first_comp'], dpl['second_comp'],
                #            color=fcolor, marker=fmarker, s=75)
                # Centroids
                ax.scatter(dcl['first_comp'], dcl['second_comp'],
                           c=fcolor, marker='X', s=125)
                
                # Plot the lines to the centroids
                for idx in range(len(dpl)):                
                    val_x = float(dpl.iloc[idx, :]['first_comp'])
                    c_x = float(dcl['first_comp'])
                    val_y = float(dpl.iloc[idx, :]['second_comp'])
                    c_y = float(dcl['second_comp'])                           
                    ax.plot([val_x, c_x], [val_y, c_y], c=fcolor, alpha=0.8)
                    
            elif style=='labels':
                ax.scatter(dpl['first_comp'], dpl['second_comp'], s=0)
                for x, y, txt in zip(dpl['first_comp'], dpl['second_comp'],
                                     dpl.index):
                    ax.text(x, y, txt, fontsize='x-small', color=fcolor)

                # Cluster number
                ax.scatter(dcl['first_comp'], dcl['second_comp'],
                           c='black', marker='$%d$' % lab,
                           s=250, edgecolor=fcolor)
                
            else: # Take care of the error message
                opt_l = ['centroids', 'labels']
                raise ValueError(f'style value should be in {opt_l}')
                    
        ax.set_title("Clustering on a Projected 2D Subspace", y=1.02)
        
        ax.set_xticks([]), ax.set_yticks([])
        if axis_label==True:
            ax.set_xlabel('First reduced dimension',
                          fontsize='small', labelpad=20)
            ax.set_ylabel('Second reduced dimension',
                          fontsize='small',labelpad=20)
        else:
            pass
            
        return(ax) # Return only the ax

###############################################################################
# Cluster Fit Performance -> Cluster Fit Performance Plots 
###############################################################################
@dataclass
class ClusterFitPerformancePlot: 
    __clusterfitperf__: ClusterFitPerformance

    def __post_init__(self): # Initialize the attributes
        self.__dict__.update(self.__clusterfitperf__.__dict__) # Import attr
    
    def silhouette(self, ax=None, **kwargs):
        """ Plot the silhouette of each cluster """

        # Arrange the data for easy plot
        dp0 = self.__clusterfitperf__.__silhouette_frame__() 
        dpg = dp0.groupby(['cluster_label'],
                          as_index=False)['silhouette_val'].mean()
        dpg = dpg.rename(columns={'silhouette_val':'silhouette_avg_cluster'})

        dp = dp0.merge(dpg, on=['cluster_label'])
        dp = dp.sort_values(by=['silhouette_avg_cluster', 'silhouette_val'],
                            ascending=[False, False])
        dp = dp.reset_index()

        # Retrieve the parameters
        sil_avg = float(dp['silhouette_avg'].unique())
        cluster_l = list(dp['cluster_label'].unique())
        len_cluster = len(cluster_l)

        # Prepare the chart
        ax = ax or plt.gca() # If None returns the other choice

        for cl in cluster_l:
            dpl = dp.loc[dp.cluster_label==cl, :].copy()
            ax.plot(dpl['silhouette_val'], dpl.index, color=self.color_d[cl],
                    alpha=0.75)
            ax.fill_betweenx(dpl.index, 0, dpl['silhouette_val'],
                             label=f'Cluster {cl}', color=self.color_d[cl],
                             alpha=0.75)

        ax.axvline(x=sil_avg, color="tab:red", linestyle="--", lw=2, 
                   label='Sample \n Average')
        ax.axvline(x=0, color="black", linestyle="-", lw=0.75)
        ax.legend(frameon=False, handlelength=1, fontsize='small')    

        ax.set_title('Separation Plot ("Silhouette" Plot)', y=1.02)
        ax.set_yticks([])

        return(ax)
       
###############################################################################
# Example
###############################################################################
if __name__ == '__main__':    
    from sklearn.datasets import make_regression     # Generate example data
    X, y = make_regression(n_samples=50, n_features=20,
                           noise=4, random_state=42)
    df = pd.DataFrame(X, columns=[f'x{idx+1}' for idx in range(X.shape[1])])
    df['label'] = [random_text(3) for _ in range(len(df))] # Add a label
    df = df.set_index('label')

    # Prepare the cluster
    clu = Cluster(df.columns, df)

    # Plot the dendogram
    fig, ax = plt.subplots()
    clu.dendogram(axvline=9, ax=ax)
    plt.show()

    # Fit the Kmeans
    clu_f = clu.fit(method='kmeans', n_clusters=5)

    # Plot on the projection on a 2D subspace with labels
    fig, ax = plt.subplots()
    clu_f.plot.cluster_2d(style='labels', ax=ax)
    plt.show()

    # Plot on the projection on a 2D subspace with centroids
    fig, ax = plt.subplots()
    clu_f.plot.cluster_2d(style='centroids', ax=ax)
    plt.show()
    
