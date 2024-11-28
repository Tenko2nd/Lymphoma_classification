<a id="readme-top"></a>



<!-- PROJECT SHIELDS -->
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT HEADER -->
<!-- <br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div> -->



<!-- TABLE OF CONTENTS -->
<!-- <details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->



<!-- ABOUT THE PROJECT -->
# About The Project

The primary goal of this project was to differentiate Mantle Cell lymphomas from Marginal Zone lymphomas, using a private dataset provided by the CHU of Dijon.</br>
I have used a pretrained CNN model to acheive it, and here I will explain you how my code works.

This project was made during a technical internship from july to november 2024.</br>
It was the collaboration between the faculty of medicine of Dijon and the start-up Ummon Healthtech.</br>
It was my first real project and I had to learn AI from scratch. 


## Built With

[![Python][Python]][Python-url]
[![Pytorch][Pytorch]][Pytorch-url]
[![scikit-learn][scikit-learn]][scikit-learn-url]



<!-- GETTING STARTED -->
# Getting Started



## Installation

1. Clone the repo
   ```sh
   git clone https://gitlab.in2p3.fr/iftim/projects/lymphoma_detection.git
   git clone https://github.com/Tenko2nd/Lymphoma_classification.git
   ```
2. Install the requirements _(due to lack of time they are nor optimised)_
   ```sh
   pip install -r requirements.txt
   ```
## One time use code
 In this part i will show you the code you need to run only once and which will create files you'll need to keep

 1. **[lymphoma_csv_data.py](./lymphoma_csv_data.py)**</br>
    For this you'll need to have the set of data given from the CHU of Dijon, and organized them like so :
    ```sh
    DS_FOLDER/     #The DS_FOLDER name is up to you
            ├── LCM/
            │   ├── patient1/           
            │   │   ├── picture1.jpg
            │   │   ├── picture2.jpg
            │   │   └── ~~~.jpg
            │   ├── patient2/
            │   └── ~~~/
            ├── LZM/
            │   ├── patient1/
            │   └── ~~~/
            └── SGTEM/
    ```
    So you need to have all your patient for each categories in their appropriate folder with those **EXACT** names (LCM, LZM, SGTEM).</br>
    _It should already be done exept for the SGTEM folder whch might be called 'témoins'_

    Then you can run the code :
    ```sh
    python /path/to/code/lymphoma_csv_data.py -d "/path/to/DS_FOLDER/"
    ```
    It have created a file _data.csv_ in your DS_FOLDER!</br> 
    _see the folder launch_code_example for other examples and the file for documentation_ 
2. **[lymphoma_dataset.py](./lymphoma_dataset.py)**</br>
    This code is here to create the dataset that will be used for training our model.</br>
    For this code, you'll need the path of the csv file we just created.</br>
    Then you'll have to enter a line of code that can have multiple parameter.
    * -csv _(mandatory)_: the path of the CSV previously created
    * -k _(optional)_: The number of fold you want for later k_fold. Defaults to 5.
    * -n _(optional)_: The number of different mapping between patient and fold. Default to 1.
    * --int : If you want internal validation (one patient is in train, val and test all at once)
    * --two : If you only want two classes (no control patients)
    ```sh
    #Examples of code

    python /path/to/code/lymphoma_dataset.py -csv "/path/to/DS_FOLDER/data.csv" -k 6 #created 1 csv with the 3classes, 6 fold, external validation
    python /path/to/code/lymphoma_dataset.py -csv "/path/to/DS_FOLDER/data.csv" -k 10 -n 2 --int --two #created 2 csv with the 2classes, 10 fold, internal validation
    ```
    It have created CSVs in a new folder _Dataset_ where the code as been launched</br> 
    _see the folder launch_code_example for other examples and the file for documentation_
3. **[label_encoder.py](./label_encoder.py)**</br>
    This code is used to create a label encoder wich will contained the classes of the dataset. It will be usefull to ensure that the classes don't get mixed up later on.</br>
    /!\ It is recommended to use a dataset containing the 3 classes to create the label encoder file.</br>
    To use this code, you only need the path to a csv dataset created in the previous part.
    ```sh
    python /path/to/code/label_encoder.py -d "/path/to/code/Dataset/DS_Lymph_3class_k5.csv" #The name depend on the csv generated before
    ```
    It have created a pickle file where the code as been launched</br>
    _see the folder launch_code_example for other examples and the file for documentation_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
# Usage

From now on, the codes used will be for training models, and testing it / generating figures out of a pre-existing model.

## Training a model

Let's start with training a model. For it, there is only the need for a single line of code.</br>
This code runs better on GPU, but if you are using a CPU that is fine to _(It will just take much more time ...)_.</br>
First, let list the option you will need for training the model :
* -d _(mandatory)_: The path to the csv **dataset** created in the code **'Lymphoma_dataset.py'**
* -bs : The batch size for the model. _default : 16_
* -lr : The learning rate for the model. _default : 0.0001_
* -wd : The weight decay for the model. _default : 0.0_
* -w : The number of workers for the model. _default : 4_
* -es : The patience value of early stopping for the model. _default : 5_
* -name : The name of your model for saving it. _default : 'anon'_
* --pb_disable : If you want to disable the progress bar
* --precomputed : If you have already precomputed the embedings ← more on that later

Even if they are not mandatory (exept the first one), I **strongly** recommend you to put your own value to try. You can use the default value as a _baseline_ for result.</br>
Now is how you run a training with this file **[lymphoma_model.py](./lymphoma_model.py)**:
```sh
#Examples of code

python /path/to/code/lymphoma_model.py -d "/path/to/code/Dataset/example.csv" -name "test" -bs 16 -w 8 -lr 0.0001 -wd 0.0001 # some option, batch size, workers, learning rate, weight decay with the name 'test'
python /path/to/code/lymphoma_model.py -d "/path/to/code/Dataset/example.csv" --precomputed # If you already have the embeddings calculated and store on the computer
```
It will run the models training, and save them in a model folder where the code has been launched. If you didn't disabled the progress bar, you will see the training in real time, else you will only see info at the end of each epoch.</br>
_see the folder launch_code_example for other examples and the file for documentation_

## Precompute the embeddings for faster computation

In the training code, we could choose if the embeddings were precomputed. It mean, the dataset has already passed the encoder, and the value given by the encoder has been stored on the computer as npy file.</br>
**There is a way to do that in this repo !!!**</br>
you just need to launch the code **[precompute_embedding.py](./precompute_embedding.py)** with this parameters :
* -d _(mandatory)_: The path to the **Global** csv created in the code **'Lymphoma_csv_data.py'**
* -bs : The batch size for the model. _default : 16_
* -w : The number of workers for the model. _default : 4_
* --pb_disable : If you want to disable the progress bar

Here is how you run it :
```sh
python /path/to/code/precompute_embedding.py -d "/path/to/DS_FOLDER/data.csv" -b 8 -w 8
```
Once this code is finished, you will have a new folder inside of your DS_FOLDER (where your images are stored) wich will contained the same architecture as the original DS_FOLDER but all the images will have been replaced by .npy files.</br>
For the moment, i only use a phikon-v2 pretrained model, so you will not be able to change wich model choose without touching the code! 

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTACT -->
# Contact

CASSERT Mathis - mathis.cassert@reseau.eseo.fr

Project Link: 
</br>[![GitHub][GitHub]][GitHub-url]
</br>[![GitLab][GitLab]][GitLab-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mathis-cassert
[Python-url]: https://www.python.org/
[Python]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Pytorch-url]: https://pytorch.org/
[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/
[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[GitHub-url]: https://github.com/Tenko2nd/Lymphoma_classification
[GitHub]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[GitLab-url]: https://gitlab.in2p3.fr/iftim/projects/lymphoma_detection
[GitLab]: https://img.shields.io/badge/gitlab-%23181717.svg?style=for-the-badge&logo=gitlab&logoColor=white