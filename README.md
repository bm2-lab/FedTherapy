# FedTherapy
A federated learning and transfer learning based multi-drug drug response prediction model.


## Introduction
In the process of constructing individualized drug response models for cancer patients, the following challenges are encountered: (1) There is a scarcity of annotated patient omics drug response data, while a substantial volume of drug perturbation data from in vitro cell lines and model organisms has accumulated in the field. Designing a sound and effective transfer learning model to integrate in vitro cell line and model organism drug perturbation data for personalized drug efficacy assessment in patients is a pressing issue in this domain; (2) Patient sample omics data carries sensitivity and privacy concerns, and as clinical research trends toward decentralization, these data are distributed across multiple nodes, necessitating the development of privacy-preserving models for integrating and analyzing patient-specific drug therapy and efficacy in a multi-center setting.

To address the aforementioned challenges, this project aims to systematically integrate in vitro cell line and model organism drug intervention data and construct an AI model, FedTherapy, for individualized precision drug therapy and efficacy assessment for cancer patients in a multi-center scenario. This involves the creation of a core dataset that encompasses cell lines, model organisms, and patient drug responses; the utilization of transfer learning algorithms to facilitate knowledge transfer from in vitro settings to clinical applications; and the application of federated learning algorithms to ensure privacy-preserving integration and utilization of patient data. The ultimate goal is to establish models for personalized drug therapy and efficacy assessment in patients.

![image](/fig/doc/all.png)


## Requirement
* python == 3.7
* pytorch == 1.4.0
* scikit-learn == 0.19.2


## Usage
**Run FedTherapy through the following steps.**
1. Preprocessing of expression profiles and drug response data.
2. Federated learning based transfer learning model pre-trainingg.
3. Fine-tuning the model with labeled source domain data.

### Preprocessing of expression profiles and drug response data
```bash
python preprocess.py
```
![image](/fig/doc/dataset.png)

### Federated learning based transfer learning model pre-training
This step simulates the process of splitting the patient domain dataset as private data and using federated learning to train the model.
Federated learning uses the FedAvg algorithm.
```bash
python Fed.py
```
![image](/fig/doc/federate.png)
![image](/fig/doc/transfer.png)

### Fine-tuning the model with labeled source domain data
Only fine tune the model on the source domain and test the model on the target domain to verify its migration ability.
```bash
python ft.py
```


## Contacts
2131453@tongji.edu.cn or qiliu@tongji.edu.cn
