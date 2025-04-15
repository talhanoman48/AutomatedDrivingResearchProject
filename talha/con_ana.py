import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

confusion_matrices = np.load("0_7.npy", allow_pickle=True)

st.write(confusion_matrices[1])

IOU=[]

ll = ["Road","Sidewalk","Parking","Tail track","Person","Rider","Car","Truck","Bus","On Rails","Motorcycle","Bicycle","Caravan","Trailer","Building","Wall","Fence","Guard Rail","Bridge","Tunnel","Pole","Traffic sign","Traffic light","Vegetation","Terrain","Sky","Ground","Dynamic","Static","Road Line","Water", "None"]

for i in range(len(confusion_matrices)):
    matrix = confusion_matrices[i]
    InOU = []
    for j in range(matrix.shape[1]):
        TP = matrix[j][j]
        FN = np.sum(matrix[:][j])
        FP = np.sum(matrix[j][:])
        if FN !=0:
            temp = (2*TP)/(FN+FP)
            InOU.append(temp)
        else:
            InOU.append(np.nan)
    IOU.append(InOU)

IOU = np.array(IOU)

st.write(IOU.shape)

mIOU = np.nanmean(IOU, axis = 1)

st.write(mIOU)

st.write(np.mean(mIOU))

avg_mIOU= np.nanmean(IOU, axis = 0)

fig = px.bar(y=avg_mIOU, x=ll, title="Average IOU over the whole dataset")

st.plotly_chart(fig)


def mIOU(numpy_filenames):
    IOUS = pd.DataFrame()

    for name in numpy_filenames:
        confusion_matrices = np.load(f"{name}.npy", allow_pickle=True)
        IOU=[]
        for i in range(len(confusion_matrices)):
            matrix = confusion_matrices[i]
            InOU = []
            for j in range(matrix.shape[1]):
                TP = matrix[j][j]
                FN = np.sum(matrix[:][j])
                FP = np.sum(matrix[j][:])
                if FN !=0:
                    temp = (TP)/(FN+FP-TP)
                    InOU.append(temp)
                else:
                    InOU.append(np.nan)
            IOU.append(InOU)

        IOU = np.array(IOU)

        st.write(IOU.shape)


        mIOU = np.nanmean(IOU, axis = 1)

        additional = pd.DataFrame({f'{name}': mIOU})

        IOUS = pd.concat([IOUS, additional], axis = 1)

    return IOUS

veh_occ = ["cloudy", "noon", "rainy", "softRain"]

df = mIOU(veh_occ)
st.write(df)

fig2 = go.Figure()

for col in df:
  fig2.add_trace(go.Box(x=df[col].values, name=df[col].name, boxpoints='outliers', jitter=0.3, pointpos=-1.8 ))
  

st.plotly_chart(fig2)

