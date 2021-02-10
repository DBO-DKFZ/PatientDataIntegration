# Combining CNN-based Histologic Whole Slide Image Analysis and Patient Data to Improve Skin Cancer Classification
Code is related to the publication "Combining CNN-based Histologic Whole Slide Image Analysis and Patient Data to Improve Skin Cancer Classification"

- patient_data_integration.ipynb: jupyter notebook that shows how the results of the publication are achieved
- training_scripts: Folder contains all scripts used for training the models of the publication
- PDI_classes_and_functions: folder contains all classes (e.g. implementations of the fusion models) and functions 

<strong>Background</strong>: Clinicians and pathologists traditionally use patient data in addition to clinical examination to support their diagnoses. 

<strong>Objectives</strong>: We investigated whether a combination of histologic whole slides image (WSI) analysis based on convolutional neural networks (CNNs) and commonly available patient data (age, sex and anatomical site of the lesion) in a binary melanoma/nevus classification task could increase the performance compared to CNNs alone.

<strong>Methods</strong>: We used 431 WSIs from two different laboratories and analysed the performance of classifiers that used the image or patient data individually or three common fusion techniques. Furthermore, we tested a naive combination of patient data and an image classifier: for cases interpreted as “uncertain” (CNN output score <0.7), the decision of the CNN was replaced by the decision of the patient data classifier.

<strong>Results</strong>: The CNN on its own achieved the best performance (mean ± standard deviation of five individual runs) with AUROC of 92.30% ± 0.23% and balanced accuracy of 83.17% ± 0.38%. While the classification performance was not significantly improved in general by any of the tested fusions, naive strategy of replacing the image classifier with the patient data classifier on slides with low output scores improved balanced accuracy to 86.72% ± 0.36%.

<strong>Conclusion</strong>: In most cases, the CNN on its own was so accurate that patient data integration did not provide any benefit.  However, incorporating patient data for lesions that were classified by the CNN with low “confidence” improved balanced accuracy.
