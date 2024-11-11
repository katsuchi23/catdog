Please download the train and test dataset and for cat and dog first before testing the code, also the final_image folder for all the image

* Train: https://drive.google.com/file/d/1hdKG5aRI3USEsa_Bxu14srTG1jJkq2gU/view?usp=sharing
* Validation: https://drive.google.com/file/d/1i3vv518YwHX7ciTlDFH8K3IU7XsyJ1D3/view?usp=sharing
* Final_Image: https://drive.google.com/file/d/12fT_NNxvRZ6fwQYAJn_0uRmI8z2TulQy/view?usp=sharing 

Additionally, dataprocessing.ipynb can be runned to produce the same result

Final_Image contains all images of model training, such as
1. CatDog-finalresult: Final model for cat and dog, which will be used to evaluate submission.csv
2. CatDog-2cnnblocks: Result of the same model as original cat and dog but with only 2cnn blocks instead of 3
3. CatDog-2cnnblocks: Result of the same model as original cat and dog but with 4cnn blocks instead of 3
4. CatDog-gridsearch: Result of all grid search for cat and dog (please refer to finetune.ipynb for more information)
5. cifer10-gridsearch: Result of all grid search for cifer10 (please refer to cifer10.ipynb for more information)
6. cifer10-testing: Result of all model architecture testing for cifer 10 (please refer to cifer 10 for more information)

Please refer to **finetune.ipynb** to see all fine tuning and model testing for cat vs dog and **cifer10.ipynb** for all mode architecture testing and fine tune for cifer10

In order to produce the same result, please run **model.py** for cat and dog, and **cifer.p**y for cifer10

Additional file:
1. Submission.csv
2. GridSearchResults.csv for all grid search result for catdog model
3. cifer10-gridsearch
