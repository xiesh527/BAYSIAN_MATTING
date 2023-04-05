# Image matting using Bayesian theorem

## Description of the project
This task is a group assignment for the course 5C22 Computational Methods. This assignment performs compositing on the image based on the image, trimap and background image. 

---


## Installation 

To install the python libraries run the below command.

```sh                                 
pip install -r requirements.txt 
```

The requiremets.txt contains the following libraries:
```sh
matplotlib==3.7.1
numba==0.56.4
numpy==1.23.5
opencv_python==4.7.0.72
Pillow==9.5.0
scikit_learn==1.2.2
scipy==1.9.1
skimage==0.19.3
```

## Execution
To run the algorithm, type the following command

```sh
python Main.py
```
To run the unit test for the program , run the following command
```sh
python unit_test.py
```
To run the e2e test for the file , run the following command
```sh
python e2e.py
```


---

## Algorithm design
The algorithm has the following steps :

1. The image, trimap, and pixels are analyzed by separating them into foreground, background, and unknown pixels.
2. The unknown pixels are categorized into clusters of foreground and background using statistical measures such as mean, covariance, and maximum eigenvalue.
3. The expectation maximization algorithm is applied to calculate the values of the foreground, background, and alpha.
4. The compositing algorithm merges the alpha matte and background to create the final image.
---
## Unit test
1. A `dim_Testing` test to validate the dimensions of the input image, trimap, alpha matte and composite image.
2. A `value_Testing` test to test the maximum and minimum value of the trimap
3. A `test_window` test to validate whether the get window function is working clearly or not.
4. A `test_Orchard_Bouman` test to validate the weight mean and covariances of the Orchard Boumann clustering algorithm.

## End-to-end testing
1. The laplacian alpha matte is generated using built in laplacian function with thresholding the pixels.
2. The laplacian alpha matte is compared with Bayesian matte.
3. The metrics such as MSE, PSNR and SSIM is calculated and the table is shown in the results section

## Results
The metrics for different images is shown.



---
## Credits

This Bayesian matting code was developed for purely academic purposes for 5C22 compuatational method assignment.
Resources:
- Marco Forte's Bayesian matting algorithm.
- Research paper - 'A Bayesian approach to Digital Matting'

---
## Consultant 
Our team got help from the other team members, we found it difficulty with uploading our code to git and create a single main repository. 

- Uditangshu Aurangabadkar, Isaiah Isijola  as Git consultant.
- ChatGPT for making it easier to understand python errors by giving simple examples.






