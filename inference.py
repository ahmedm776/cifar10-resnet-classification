"""CIFAR-10 ResNet inference - Stage E best model (epoch 4, val-selected).
Usage:
  python3 inference.py --images path/to/images_u8.npy [--labels path/to/labels.npy]
  python3 inference.py --selftest
images_u8.npy: uint8 array (N,32,32,3), CIFAR-10 layout. Preprocessing: resize to 64x64, scale /255.
Classes: airplane automobile bird cat deer dog frog horse ship truck
"""
import argparse, json
import numpy as np
import tensorflow as tf, keras
CLASSES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def load():
    return keras.models.load_model("model_best.keras")
def predict_proba(model,x_u8,batch=64):
    ds=tf.data.Dataset.from_tensor_slices(x_u8)
    ds=ds.batch(batch).map(lambda a: tf.image.resize(a,(64,64))/255.0,num_parallel_calls=2).prefetch(2)
    return model.predict(ds,verbose=0)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--labels"); ap.add_argument("--selftest",action="store_true")
    a=ap.parse_args()
    m=load()
    if a.selftest:
        x=np.load("sample_images_u8.npy"); exp=np.load("sample_expected_probs.npy")
        p=predict_proba(m,x)
        ok=np.allclose(p,exp,atol=1e-5)
        acc=float((p.argmax(1)==np.load("sample_labels.npy")).mean())
        print(json.dumps({"selftest_probs_match":bool(ok),"max_abs_diff":float(np.abs(p-exp).max()),"sample_acc":acc}))
        return
    x=np.load(a.images); p=predict_proba(m,x)
    out={"preds":[CLASSES[i] for i in p.argmax(1)],"probs":p.tolist()}
    if a.labels: out["acc"]=float((p.argmax(1)==np.load(a.labels).reshape(-1)).mean())
    print(json.dumps(out))
if __name__=="__main__": main()
