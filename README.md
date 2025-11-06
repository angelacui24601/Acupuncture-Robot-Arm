# Simultaneous Multimodal Detection of Hand Acupoints and Reflex Zones for Acupuncture Robots
This project is the official code implementation of the paper 《Simultaneous Multimodal Detection of Hand Acupoints and Reflex Zones for Acupuncture Robots》. This warehouse provides the complete implementation of the MIMO-HAR framework, which is a novel multimodal and multi-task deep learning framework designed for intelligent acupuncture robots.

## 1. Model framework
The MIMO-HAR framework is based on the Transformer architecture and consists of four key modules: ViT image encoder, prior acupoint input module, Class-Wise Mask Decoder and Class-Wise Point Decoder.
![image.png](https://raw.gitcode.com/user-images/assets/7559531/2eda9321-8846-40ed-8901-32bdea0c731c/image.png 'image.png')

## 2 Dataset

### 2.1 Basic Image Dataset

We used the publicly available "11k Hands" dataset as the base image source. This dataset contains 11,076 hand images from 190 subjects.
Open access: you can be found at https://github.com/mahmoudnafifi/11K-Hands.

### 2.2 MIMO-HAR Dataset

Note: The "11k Hands" dataset was originally used for gender identification and biometric recognition and does not include the labeling of acupoints or reflex zones. One of the core contributions of this project is the creation of a brand-new, high-quality expert annotation set. We selected 1,000 images from the "11k Hands" dataset and manually annotated them using the Labelme tool by a team led by senior traditional Chinese medicine experts.

Quality control: The annotation process underwent a strict consistency assessment. The average Dice coefficient of the reflex zone was 0.89, and the average normalized coordinate distance of the acupoints was 0.04, ensuring a high degree of consistency among annotators.

### 2.3 Data Request

The code in this repository is designed for using our specific expert annotation format. The basic 11k Hands image dataset is public, but the new high-precision annotation files we created for 16 acupoints and 14 reflex zones are important assets of our research team and are not included in the public release.

We welcome academic and research cooperation. If you need to obtain annotation files for academic research, please contact us. email: qh2020@jhun.edu.cn.

## 3 Environmental Installation

Step 1: Clone the repository
		
        git clone https://gitcode.com/qq_38063965/AcupointMMNet_Hand.git
    	cd AcupointMMNet_Hand
        
Step 2: Create and activate the Conda environment

	conda create -n acupointnet python=3.7 -y
	conda activate acupointnet

Step 3: Installation Dependencies

	pip install -r requirements.txt

Step 4: Training and Testing

	sh ./train.sh
    python test.py
Note: Please modify the .sh script as required before running

## 4. Cite
If you have used this project or the MIMO-HAR framework in your research, please cite our paper:

Zheng, Y., Liao, C., Zhang, H., & He, Q. (2025). Simultaneous Multimodal Detection of Hand Acupoints and Reflex Zones for Acupuncture Robots. 