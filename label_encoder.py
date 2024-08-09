"""
    This python code is used to create a <Label Encoder> based on the categories of the dataset created, thanks to the code "lymphoma_dataset.py".\n
    It take the different classes defined in the <classes> intern variable of the dataset and fit them to the label encoder\n
    //!\\ The classes must be references in a variable named 'classes' inside of the __init__ function of the dataset. Else replace the name with your own.\n
    After that the different classes are serialized in a <Pickle> file with the name "Lymphoma_labelEncoder.pkl".\n
    //!\\ To deserialize the label encoder and use it you must follow this steps :\n
        from sklearn import preprocessing                   \n
        with open('Lymphoma_labelEncoder.pkl', 'rb') as f:  \n
            le = preprocessing.LabelEncoder()               \n
            le.classes_ = pickle.load(f)                    \n

"""

from lymphoma_dataset import LymphomaDS_resize360
from sklearn import preprocessing
import pickle


le = preprocessing.LabelEncoder()
le.fit(LymphomaDS_resize360.classes)

with open("Lymphoma_labelEncoder.pkl", "wb") as f:
    pickle.dump(le.classes_, f)
