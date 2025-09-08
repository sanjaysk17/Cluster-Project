import gradio as gr
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering,DBSCAN
df=pd.read_csv("data_student_synthetic.csv")
x=df.drop(" UNS",axis=1)
scaler=pickle.load(open("scaler.pkl","rb"))
x_scaled=scaler.fit_transform(x)
modelk=pickle.load(open("modelk.pkl","rb"))
def predict(STG,SCG,STR,LPR,PEG,algorithm):
    new_point=np.array([[STG,SCG,STR,LPR,PEG]])
    new_points=scaler.transform(new_point)
    if algorithm=="KMeans":
        output=modelk.predict(new_point)[0]
        return f"KMeans:Cluster {output}"
    elif algorithm=="Hierarchical":
        x_temp=np.vstack([x_scaled,new_points])
        modela=AgglomerativeClustering(n_clusters=2,metric="euclidean",linkage="ward")
        labels=modela.fit_predict(x_temp)
        return f"Hierarchical: Cluster {labels[-1]}"
    elif algorithm=="DBSCAN":
        x_temp=np.vstack([x_scaled,new_points])
        modeld=DBSCAN(eps=0.5,min_samples=5)
        labels=modeld.fit_predict(x_temp)
        if labels[-1]==-1:
            return "DBSCAN: Noise"
        else:
            return f"DBSCAN: Cluster {labels[-1]}"
    else:
        return "Unknown algorithm"
USERNAME="sanjay"
PASSWORD="1234"
def check_login(username,password):
    if username==USERNAME and password==PASSWORD:
        return gr.update(visible=False),gr.update(visible=True),"✅ Login successful!"
    else:
        return gr.update(visible=True),gr.update(visible=False),"❌ Invalid credentials"
with gr.Blocks() as demo:
    with gr.Row(visible=True) as login_page:
        gr.Markdown("## 🔑 Login Page")
        user=gr.Textbox(label="Username")
        pwd=gr.Textbox(label="Password",type="password")
        login_btn=gr.Button("Login")
        login_status=gr.Label()
    with gr.Row(visible=False) as app_page:
        interface=gr.Interface(fn=predict,
            inputs=[gr.Number(label="STG"),
                gr.Number(label="SCG"),
                gr.Number(label="STR"),
                gr.Number(label="LPR"),
                gr.Number(label="PEG"),
                gr.Dropdown(choices=["KMeans","Hierarchical","DBSCAN"],label="Select Algorithm")],
            outputs="text",
            title="Clustering",
            description="Enter values for STG, SCG, STR, LPR, PEG and choose a clustering algorithm",
            live=False)
        interface.render()
    login_btn.click(fn=check_login,inputs=[user,pwd],outputs=[login_page,app_page,login_status])
if __name__=="__main__":
    demo.launch()
