"""
Generate one model for all wells.

Designed to:
    read well data,

python seisattrib2log_build.py -h
python seisattrib2log_build.py allsgy.txt allgr.csv --startendinterval 500 1000
python seisattrib2log_build.py limbayongattrib.txt allgr.csv --startendinterval 1400 2900 --intime
"""

import os.path
import argparse
# from datetime import datetime
import numpy as np
import scipy.signal as sg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as sts
import segyio as sg
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score

try:
    from catboost import CatBoostRegressor
except ImportError:
    print('***Warning:CatBoost is not installed')

def pip(x, y, poly):
    # check if point is a vertex
    """
    if (x,y) in poly:
        return True
    """
    # check if point is on a boundary
    for i in range(len(poly)):
        p1 = None
        p2 = None
        if i == 0:
            p1 = poly[0]
            p2 = poly[1]
        else:
            p1 = poly[i-1]
            p2 = poly[i]
        if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
            return True

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    if inside: return True
    else: return False

#...............................................................................
class Invdisttree:

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]


def idw(xy,vr,xyi):

    # N = vr.size
    # Ndim = 2
    # Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 2  # weights ~ 1 / distance**p
    invdisttree = Invdisttree( xy, vr, leafsize=leafsize, stat=1 )
    interpol = invdisttree( xyi, nnear=Nnear, eps=eps, p=p )
    return interpol

def qhull(sample):
    link = lambda a,b: np.concatenate((a,b[1:]))
    edge = lambda a,b: np.concatenate(([a],[b]))
    def dome(sample,base):
        h, t = base
        dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
        outer = np.repeat(sample, dists>0, 0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:,0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], 0)
        return link(dome(sample, base),dome(sample, base[::-1]))
    else:
        return sample

def gridlistin(fname,xyvcols=[0,1,2],nheader=0): #used for single coef per file
    xyv=np.genfromtxt(fname,usecols=xyvcols,skip_header=nheader)
    #filter surfer null values by taking all less than 10000, arbitrary!!
    xyv = xyv[xyv[:,2]<10000.0]
    #xya = xya[~xya[:,2]==  missing]
    return xyv[:,0],xyv[:,1],xyv[:,2]


def map2ddata(xy,vr,xyi,radius=5000.0,maptype='idw'):
    stats=sts.describe(vr)
    # statsstd=sts.tstd(vr)
    if maptype == 'idw':
        vri=idw(xy,vr,xyi)
    elif maptype =='nearest':
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='nearest')
    elif maptype == 'linear':
        #                vri=griddata(xy,vr,(xyifhull[:,0],xyifhull[:,1]),method='linear')
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='linear')
    elif maptype == 'cubic':
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='cubic')
    elif maptype =='rbf':
        rbf=Rbf(xy[:,0],xy[:,1],vr)
        vri= rbf(xyi[:,0],xyi[:,1])
    # elif maptype =='avgmap':
    #     vri=dataavgmap(xy,vr,xyi,radius)
    elif maptype =='triang':
        linearnd=LinearNDInterpolator(xy,vr,stats[2])
        vri= linearnd(xyi)
    elif maptype == 'ct':
        ct=CloughTocher2DInterpolator(xy,vr,stats[2])
        vri=ct(xyi)
    return vri.astype(int)


def filterhullpolygon(x,y,polygon):
    xf=[]
    yf=[]
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            xf.append(x[i])
            yf.append(y[i])
    return np.array(xf),np.array(yf)

def filterhullpolygon_mask(x,y,polygon):
    ma=[]
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            ma.append(True)
        else:
            ma.append(False)
    return np.array(ma)



def process_segylist(segyflist):
    """Process list of segy attribute files."""
    segylist = []
    with open(segyflist,'r') as f:
        for line in f:
            segylist.append(line.rstrip())
    return segylist

def get_slice(fname,slicenum):
    slc = list()
    with sg.open(fname,'r',ignore_geometry= True) as srcp:
        for trnum,tr in enumerate(srcp.trace):
            slc.append(tr[slicenum])
    return slc

def get_onetrace(fname,tracenum,sstart=None,send=None):
    """Get one trace from one file."""
    with sg.open(fname,'r',ignore_geometry=True) as srcp:
        tri = srcp.trace[tracenum]
        yield tri[sstart:send]

def collect_traces(segyflist,numoftraces,trcnum,sstart=None,send=None):
    """Collect the same trace from all attribute segy files."""
    for trn in range(numoftraces):
        trclst = list()
        for fn in segyflist:
            trclst.append(get_onetrace(fn,trn,sstart,send))
        trca = np.array(trclst)
        Xpred = trca.T
        # trace array transposed
        yield Xpred


def zero_segy(fname):
    """."""
    with sg.open(fname,'r+',ignore_geometry=True) as srcp:
        for trnum,tr in enumerate(srcp.trace):
            srcp.trace[trnum] = tr * 0

def get_samplerate(fname):
    """."""
    with sg.open(fname,'r',ignore_geometry=True) as srcp:
        hdrdict = dict(enumerate(srcp.header[1].items()))
    return hdrdict[39][1] / 1000

def get_xy(fname,xhdr,yhdr,xyscalerhdr,trlstevery=50000):
    """."""
    xclst = list()
    yclst = list()
    trnlst =list()
    with sg.open(fname,'r',ignore_geometry=True) as srcp:
        print(f'Total # of Traces: {len(srcp.trace)}')
        for trnum,tr in enumerate(srcp.trace):
            xysch = np.fabs(srcp.header[trnum][xyscalerhdr])
            if xysch == 0:
                xysc = 1.0
            else:
                xysc = xysch
            xci = srcp.header[trnum][xhdr] / xysc
            yci = srcp.header[trnum][yhdr] / xysc
            xclst.append(xci)
            yclst.append(yci)
            trnlst.append(trnum)
            if trnum % trlstevery == 0:
                print(f'Trace # {trnum:0d}  {xci:.2f}  {yci:.2f}')
    xc = np.array(xclst)
    yc = np.array(yclst)
    trn = np.array(trnlst,dtype=int)
    trcols = ['XC','YC','TRNUM']
    xytrcdf = pd.DataFrame({'XC':xc,'YC':yc,'TRNUM':trn})
    xytrcdf = xytrcdf[trcols].copy()
    print(xytrcdf.head())
    return xytrcdf

def process_xytrace(sa,xw,yw):
    """.
    sa is a df of x y tr from headers
    xtr,ytr are x y coordinates of list of traces to locate trace numbers
    """
    # print(sa.head())
    xs = sa.iloc[:,0]
    ys = sa.iloc[:,1]
    xys = np.transpose(np.vstack((xs,ys)))
    xyw = np.transpose(np.vstack((xw,yw)))
    # tr_atxy = map2ddata(xys,sa.iloc[:,2],xyw,maptype='cubic')
    tr_atxy = map2ddata(xys,sa.iloc[:,2],xyw,maptype='triang')

    return tr_atxy

def seisattrib_atwell(sflist,swa):
    sr = get_samplerate(sflist[0])
    swa['SLICENUM'] = swa.iloc[:,1] // sr
    for sf in sflist:
        print(sf)
        dirsplit,fextsplit= os.path.split(sf)
        fname,fextn= os.path.splitext(fextsplit)
        tracesample=list()
        with sg.open(sf,'r',ignore_geometry=True) as srcp:
            for i in range(swa.shape[0]):
                tracesample.append(srcp.trace[swa.loc[i,'TRNUM']][swa.loc[i,'SLICENUM']  ])
            swa[fname] = tracesample
    swa.drop(['TRNUM','SLICENUM'],inplace=True,axis=1)
    colslst =swa.columns.tolist()
    colslst.append(colslst[4])
    colslst.pop(4)
    print(colslst)
    swa = swa[colslst]
    return swa

def process_swscalecols(wseisdf,includexy=False):
    """."""
    if includexy:
        wseisdf['Xscaled'] = wseisdf[wseisdf.columns[0]]
        wseisdf['Yscaled'] = wseisdf[wseisdf.columns[1]]
        # seisdf['Zscaled'] = seisdf[seisdf.columns[2]]
        # z is not added because it is a slice, i.e. constant z
    # print(f'wseisdf before: {wseisdf.shape}')
    if wseisdf.isnull().values.any():
        print(f'Warning: Null Values in the file will be dropped')
        wseisdf.dropna(inplace=True)
    # print(f'wseisdf after: {wseisdf.shape}')

    wseisdfcols = wseisdf.columns.tolist()
    attribonly = wseisdf.iloc[:,4:-1]
    # print(attribonly.head())

    attribcols = attribonly.columns.tolist()
    attribonlyscaled = StandardScaler().fit_transform(attribonly.values)
    # scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    attribonlyscaleddf = pd.DataFrame(attribonlyscaled,columns = attribcols,index=wseisdf.index)
    # print(attribonlyscaleddf.head())
    wseisdfsxyz = pd.concat([wseisdf.iloc[:,:4],attribonlyscaleddf,wseisdf.iloc[:,-1]],axis=1)

    wseisdfsxyz = wseisdfsxyz[wseisdfcols]
    print(wseisdfsxyz.head())
    return wseisdfsxyz

def process_seiswellattrib(sa,wa,intime):
    """."""
    # print(sa.head())
    # print(wa.head())
    xs = sa.iloc[:,0]
    ys = sa.iloc[:,1]
    xys = np.transpose(np.vstack((xs,ys)))
    xyhull = qhull(xys)

    xw = wa.iloc[:,2].values
    yw = wa.iloc[:,3].values
    # xyw = np.transpose(np.vstack((xw,yw)))
    wz = wa.iloc[:,1]
    # z value
    wid = wa.iloc[:,0]
    # wellname
    wida = wa.iloc[:,4]
    # well porosity or any other attribute

    ma = filterhullpolygon_mask(xw,yw,xyhull)
    # print('Remaining Wells after convex hull: ',len(ma))
    xwf= xw[ma]
    ywf = yw[ma]
    wzf = wz[ma]
    widf = wid[ma]
    wattribf = wida[ma]

    xywf = np.transpose(np.vstack((xwf,ywf)))
    if intime:
        welldfcols =['WELL','TIME','DEVX','DEVY']
    else:
        welldfcols =['WELL','DEPTH','DEVX','DEVY']

    # wellsin_df = pd.DataFrame([widf,xwf,ywf,wattrib],columns=welldfcols)
    wellsin_df = pd.DataFrame(widf,columns = [welldfcols[0]])
    wellsin_df[welldfcols[1]] = wzf
    wellsin_df[welldfcols[2]] = xwf
    wellsin_df[welldfcols[3]] = ywf
    # print('wellsin df shape:',wellsin_df.shape)
    # print(wellsin_df.head(10))

    nattrib = sa.shape[1]
    # print('nattrib:',nattrib)
    # welldflist =[]
    for i in range(2,nattrib):
        vs = sa.iloc[:,i].values
        zwsa = map2ddata(xys,vs,xywf)
        # print('i:',i,'zwsa:',zwsa.size)
        # welldflist.append(zwsa)
        colname = sa.columns[i]
        wellsin_df[colname] = zwsa
    wa_col = wa.columns[4]
    wellsin_df[wa_col] = wattribf
    # print('Inside seiswellattrib ....')
    # print(wellsin_df.tail())
    # print('wellsin shape:',  wellsin_df.shape)
    wellsin_df.dropna(axis=0,inplace=True)
    # print(wellsin_df.tail())
    # print('wellsin shape after dropna:',  wellsin_df.shape)

    return wellsin_df

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: n
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y



def plotwells(wdf,hideplots=True):
    """."""
    wlst = wdf[wdf.columns[0]].unique().tolist()
    print(wlst)
    pdfcl = "AllWellsPsuedoLogs.pdf"
    with PdfPages(pdfcl) as pdf:
        for wname in wlst:
            wndf = wdf[wdf[wdf.columns[0]] == wname]
            fig,ax = plt.subplots(figsize=(4,7))
            ax.invert_yaxis()
            ax.plot(wndf[wndf.columns[-2]],wndf[wndf.columns[1]],label='Actual')
            ax.plot(wndf[wndf.columns[-1]],wndf[wndf.columns[1]],label='Predicted')
            ax.set_xlabel(wndf.columns[-1])
            ax.set_ylabel(wndf.columns[1])
            ax.set_title(wname)
            ax.legend()
            fig.tight_layout()
            pdf.savefig()
            if not hideplots:
                fig.show()
            plt.close()

    pdfclh = "AllWellshistograms.pdf"
    with PdfPages(pdfclh) as pdf:
        for wname in wlst:
            wndf = wdf[wdf[wdf.columns[0]] == wname]
            plt.figure(figsize=(6,6))
            sns.distplot(wndf.GR,kde=True,
                hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "m","label": "GR"},
                kde_kws={"color": "m", "lw": 2, "label": "GR"})
            sns.distplot(wndf.GRPRED,kde=True,
                hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "g","label": "GRPRED"},
                kde_kws={"color": "g", "lw": 2, "label": "GRPRED"})
            plt.title(wname)
            pdf.savefig()
            if not hideplots:
                plt.show()
            plt.close()

    pdfecdf = "AllWellsecdf.pdf"
    with PdfPages(pdfecdf) as pdf:
        for wname in wlst:
            wndf = wdf[wdf[wdf.columns[0]] == wname]
            plt.figure(figsize=(6,6))
            fig,ax = plt.subplots(figsize=(8,6))
            grx,gry = ecdf(wndf.GR)
            ax.scatter(grx,gry,s=5,c='m',label='Actual')
            grpredx,grpredy = ecdf(wndf.GRPRED)
            ax.scatter(grpredx,grpredy,s=5,c='g',label='Predicted')
            ax.legend()
            plt.title(wname)
            pdf.savefig()
            if not hideplots:
                plt.show()
            plt.close()



def gensamples(datain,targetin,
    ncomponents=2,
    nsamples=10,
    kind='r',
    func=None):
    """
    Generate data using GMM.

    Reads in features array and target column
    uses GMM to generate samples after scaling target col
    scales target col back to original values
    saves data to csv file
    returns augmented data features array and target col

    newfeatures,newtarget =gensamples(wf,codecol,kind='c',func='cbr')
    """
    d0 = datain
    t0 = targetin
    d0t0 = np.concatenate((d0,t0.values.reshape(1,-1).T),axis=1)
    sc = StandardScaler()
    t0min,t0max = t0.min(),t0.max()
    t0s = sc.fit_transform(t0.values.reshape(1,-1))
    d0t0s = np.concatenate((d0,t0s.T),axis=1)
    gmm = mixture.GaussianMixture(n_components=ncomponents,covariance_type='spherical', max_iter=500, random_state=0)
    gmm.fit(d0t0s)
    d0sampled = gmm.sample(nsamples)[0]
    d1sampled = d0sampled[:,:-1]
    targetunscaled = d0sampled[:,-1].reshape(1,-1)
    scmm = MinMaxScaler((t0min,t0max))
    if kind == 'c':
        targetscaledback = np.floor(scmm.fit_transform(targetunscaled.T))
    else:
        targetscaledback = scmm.fit_transform(targetunscaled.T)
    d1t1 = np.concatenate((d1sampled,targetscaledback),axis=1)
    d1 = np.concatenate((d0t0,d1t1))
    print(d1.shape)
    fname = 'gensamples_' + func + '.csv'
    np.savetxt(fname,d1,fmt='%.3f',delimiter=',')
    return d1[:,:-1],d1[:,-1]

def modelpredict(model,Xpred,fromfile=False,scalelog=True,logmin=0,logmax=150):
    """."""
    if fromfile:
        modelin = CatBoostRegressor()
        # Load the model from JSON
        modelin.load_model(model)
    else:
        modelin = model
    ypred = modelin.predict(Xpred)
    # grmin,grmax = wdf.iloc[:,gr].min(),wdf.iloc[:,gr].max()
    if scalelog:
        sc = MinMaxScaler((logmin,logmax))
        # wdf['LIsc'] = sc.fit_transform(wdf['LI'].values.reshape(-1,1))
        ypredsc = sc.fit_transform(ypred.reshape(-1,1))
        return ypredsc
    else:
        return ypred

def model_xvalidate(wdfsa,
    cv=5,
    cbrlearningrate=None,
    cbriterations=None,
    cbrdepth=None,
    generatesamples=False,
    generatensamples=10,
    generatencomponents=2):
    """Cross validate CBR."""
    X = wdfsa.iloc[:,4: -1]
    y = wdfsa.iloc[:,-1]
    if y.size > 2 and generatesamples:
        X,y = gensamples(X,y,
            nsamples=generatensamples,
            ncomponents=generatencomponents,
            kind='r',func='cbr')

    model = CatBoostRegressor(iterations=cbriterations,
        learning_rate=cbrlearningrate,
        depth=cbrdepth,
        loss_function='RMSE',
        random_seed=42,
        logging_level='Silent')

    cvscore = cross_val_score(model,X,y,cv=cv,scoring='r2')
    # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
    print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
    # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
    cvscore = cross_val_score(model,X,y,cv=cv,scoring='neg_mean_squared_error')
    print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))

    # Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=valsize,
    #         random_state=42)
    # model.fit(Xtrain, ytrain)
    # print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))

    # yvalpred = model.predict(Xval)

    # # Calculating Mean Squared Error
    # msev = np.mean((yvalpred - yval)**2)
    # print('Metrics on Train-Test-Split data: ')
    # print('Train-Test-Split MSE: %.4f' % (msev))
    # r2v = r2_score(yval,yvalpred)
    # print('Train-Test-Split R2 : %10.3f' % r2v)
    # ccxv = sts.pearsonr(yval,yvalpred)
    # print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccxv[0],ccxv[1]))

def model_create(wdfsa,
    cbrlearningrate=None,
    cbriterations=None,
    cbrdepth=None,
    generatesamples=False,
    generatensamples=10,
    generatencomponents=2,savemodelname=None):
    """Read in welldata and generate model."""
    X = wdfsa.iloc[:,4: -1]
    y = wdfsa.iloc[:,-1]
    if y.size > 2 and generatesamples:
        X,y = gensamples(X,y,
            nsamples=generatensamples,
            ncomponents=generatencomponents,
            kind='r',func='cbr')
    model = CatBoostRegressor(iterations=cbriterations,
        learning_rate=cbrlearningrate,
        depth=cbrdepth,
        loss_function='RMSE',
        random_seed=42,
        logging_level='Silent')
    model.fit(X, y)

    model.save_model(savemodelname,format="json",export_parameters=None)
    print(f'Successfully saved CatBoostRegressor Model {savemodelname}')

    return model


def apply_model_towells(allwdfsa,cbrmodel,
    logname=None,scalelog=True,
    generatesamples=False,
    generatensamples=10,
    generatencomponents=2,
    plotfname=None,
    dirsplit=None,
    hideplots=False,
    outdir=None):
    """apply previously generated model."""
    X = allwdfsa.iloc[:,4 : -1]
    y = allwdfsa.iloc[:,-1]
    inshape = y.size
    # print( f"size of y: {inshape}")
    if y.size > 2 and generatesamples:
        X,y = gensamples(X,y,
            nsamples=generatensamples,
            ncomponents=generatencomponents,
            kind='r',func='cbr')
    # Get predictions
    ypred = cbrmodel.predict(X)
    if scalelog:
        logmin = allwdfsa.iloc[:,-1].min()
        logmax = allwdfsa.iloc[:,-1].max()
        print(logmin,logmax)
        sc = MinMaxScaler((logmin,logmax))
        ypredsc = sc.fit_transform(ypred.reshape(-1,1))
        ypred = ypredsc.reshape(-1)
    predcolname = logname +'PRED'
    allwdfsa[predcolname] = ypred
    # Calculating Mean Squared Error
    mse = np.mean((ypred - y)**2)
    print('Metrics on input data: ')
    print('MSE: %.4f' %(mse))
    r2 = r2_score(y,ypred)
    print('R2 : %10.3f' % r2)

    ccmdl = sts.pearsonr(y,ypred)
    qc0 = np.polyfit(y,ypred,1)
    xrngmin,xrngmax = y.min(),y.max()
    xvi = np.linspace(xrngmin,xrngmax)
    yvi0 = np.polyval(qc0,xvi)

    fig,ax = plt.subplots()

    plt.scatter(y,ypred,alpha=0.5,c='b',s=15,label='Model Predicted')
    if generatesamples:
        ax.scatter(y[inshape:],ypred[inshape:],c='r', marker='X',s=25,label='Generated Samples')

    plt.plot(xvi,yvi0,c='k',lw=2)



    ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy =(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
    ax.annotate ('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy =(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
    ax.set_title(f'CBR for all Pseudo {logname}' )
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    fig.savefig(plotfname)
    if not hideplots:
        plt.show()
    return allwdfsa


def getcommandline():
    """Get command line options."""
    parser = argparse.ArgumentParser(description='Build one ML model to convert seismic to logs')
    parser.add_argument('segyfileslist',help='File that lists all segy files to be used for attributes')
    parser.add_argument('wellscsv',help='csv file with well depth devx devy log ')
    parser.add_argument('--segyxhdr',type=int,default=73,help='xcoord header.default=73')
    parser.add_argument('--segyyhdr',type=int,default=77,help='ycoord header. default=77')
    parser.add_argument('--xyscalerhdr',type=int,default=71,help='hdr of xy scaler to divide by.default=71')
    parser.add_argument('--startendinterval',type=int,nargs=2,default=[1000,2000],
        help='Start end interval in depth/time. default= 1000 2000')
    parser.add_argument('--cbriterations',type=int,default=500,help='Learning Iterations, default =500')
    parser.add_argument('--cbrlearningrate',type=float,default=0.01,help='learning_rate. default=0.01')
    parser.add_argument('--cbrdepth',type=int,default=6,help='depth of trees. default=6')
    parser.add_argument('--includexy',action='store_true',default=False,
        help='include x y coords in model.default= not to')
    parser.add_argument('--slicesout',action='store_true',default=False,
        help='Save individual unscaled slices to csv. default=false, i.e do not save')

    parser.add_argument('--generatesamples',action='store_true',default=False,help='Generate Samples.default=False')
    parser.add_argument('--generatensamples',type=int,default=10,help='# of sample to generate. default= 10')
    parser.add_argument('--generatencomponents',type=int,default=2,help='# of clusters for GMM.default=2')
    parser.add_argument('--intime',action='store_true',default=False,
        help='processing domain. default= True for depth')
    parser.add_argument('--scalelog',action='store_false',default=True,
        help='do not apply min max scaler to computed Psuedo GR. default apply scaling')
    # parser.add_argument('--logscalemm',nargs=2,type=float,default=(0,150),
    #     help='Min Max values to scale output computed Psuedo GR trace. default 0 150')

    parser.add_argument('--outdir',help='output directory,default= same dir as input')
    parser.add_argument('--hideplots',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    result = parser.parse_args()
    if not result.segyfileslist:
        parser.print_help()
        exit()
    else:
        return result



def main():
    """main program."""
    cmdl = getcommandline()
    if cmdl.wellscsv:
        allwells = pd.read_csv(cmdl.wellscsv)
        # dz = np.diff(allwells.DEPTH)[2]
        dz = np.diff(allwells[allwells.columns[1]])[2]
        print('Well Vertical increment {}'.format(dz))
        wdirsplit,wfextsplit = os.path.split(cmdl.wellscsv)
        wfname,wfextn = os.path.splitext(wfextsplit)
        # logname = allwells.columns[-1]
        wcols = allwells.columns.tolist()
        print(wcols)
        logname = wcols[-1]
        print('logname:',logname)
        lognamepred = logname + 'pred'
        wcols.append(lognamepred)
    if cmdl.segyfileslist:
        sflist = list()
        sflist = process_segylist(cmdl.segyfileslist)
        print(sflist)
        alldirsplit,allfextsplit = os.path.split(cmdl.segyfileslist)
        allfname,allfextn = os.path.splitext(allfextsplit)
        # allfname for naming cbr model
        dirsplit,fextsplit = os.path.split(sflist[0])
        fname,fextn = os.path.splitext(fextsplit)

        sr = get_samplerate(sflist[0])
        print('Seismic Sample Rate: {}'.format(sr))

        xytrcdf = get_xy(fextsplit,cmdl.segyxhdr,cmdl.segyyhdr,cmdl.xyscalerhdr)
        swfname = 'SWAttrib'
        if cmdl.outdir:
            wsdf = os.path.join(cmdl.outdir,swfname) + f"_{logname}.csv"
            wsdfpred = os.path.join(cmdl.outdir,swfname) + f"_{logname}_pred.csv"
            pdfxplot = os.path.join(cmdl.outdir,logname) + 'xplt.pdf'
            pdfwplot = os.path.join(cmdl.outdir,logname) + 'wplt.pdf'
        else:
            wsdf = os.path.join(dirsplit,swfname) + f"_{logname}.csv"
            wsdfpred = os.path.join(dirsplit,swfname) + f"_{logname}_pred.csv"
            pdfxplot = os.path.join(dirsplit,logname) + 'xplt.pdf'
            pdfwplot = os.path.join(dirsplit,logname) + 'wplt.pdf'

        allwells['TRNUM'] = process_xytrace(xytrcdf,allwells.iloc[:,2],allwells.iloc[:,3])
        print('Allwells df:')
        print(allwells.head())
        fig,ax = plt.subplots()
        p = ax.scatter(allwells.DEVX,allwells.DEVY,c=allwells.TRNUM,cmap='rainbow',s=5)
        fig.colorbar(p)
        plt.show()
        fig.savefig(pdfwplot)
        if not cmdl.hideplots:
            plt.show()

        allwdfa = seisattrib_atwell(sflist,allwells)
        allwdfsa = process_swscalecols(allwdfa)
        allwdfsa.head()
        allwdfsa.to_csv(wsdf,index=False)
        print(f'Successfully generated all attributes with wells {wsdf}')

        cbrmodelname = allfname + '_cbr.json'

        cbrmodel = model_create(allwdfsa,savemodelname=cbrmodelname,
            cbrlearningrate=cmdl.cbrlearningrate,
            cbriterations=cmdl.cbriterations,
            cbrdepth=cmdl.cbrdepth)
        allwdfsapred = apply_model_towells(allwdfsa,cbrmodel,
            scalelog=cmdl.scalelog,
            logname=logname,
            dirsplit=dirsplit,
            plotfname=pdfxplot,
            hideplots=cmdl.hideplots,
            outdir=cmdl.outdir)
        allwdfsapred.to_csv(wsdfpred,index=False)
        print(f'Successfully generated all attributes with wells prediction {wsdfpred}')
        plotwells(allwdfsapred)
if __name__ == '__main__':
    main()
