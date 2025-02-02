<h2>Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel (2025/02/02)</h2>

This is the first experiment of Tiled Image Segmentation for <b>STARE Retinal Vessel</b>
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a <b>pre-augmented tiled dataset</b> <a href="https://drive.google.com/file/d/18PvKykorWlO-dnZq9njCU0kVCa24w8Us/view?usp=sharing">
Augmented-Tiled-STARE-ImageMask-Dataset.zip</a>, which was derived by us from the following images and labels:<br><br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar">
<b>
Twenty images used for experiments
</b>
</a>
<br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar">
<b>
Hand labeled vessel network provided by Adam Hoover
</b>
</a>
<br>
<br>
On detail of <b>STARE(STructured Analysis of the Retina)</b>, 
please refer to the official site:<br>
<a href="https://cecas.clemson.edu/~ahoover/stare/">
STructured Analysis of the Retina
</a>
, and github repository <a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md">
STARE
</a>
<br><br>
Please see also our experiment  
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
Tensorflow-Image-Segmentation-Retinal-Vessel</a> based on <a href="https://researchdata.kingston.ac.uk/96/">CHASE_DB1 dataset</a>.
<br>
<br>
<b>Experiment Strategies</b><br>
As demonstrated in our experiments <a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-IDRiD-HardExudates">
Tensorflow-Tiled-Image-Segmentation-IDRiD-HardExudates </a>, 
and <a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer">
Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer </a>, the Tiled Image Segmentation based on a simple UNet model trained by a tiledly-splitted images and masks dataset, 
is an effective method for the large image segmentation over 4K pixels.
<br><br>
It is difficult to precisely segment Retinal Blood Vessels in small images using a simple UNet model 
because these vessels are typically very thin and difficult to detect. 
Therefore, we generate a high-resolution retinal 
image dataset by upscaling the original images and use it to train the UNet model to improve segmentation performance.
<br>
<br>
In this experiment, we employed the following strategies to apply the Tiled Image Segmentation method to STARE Retinal Vessel.
<br>
<b>1. Enlarged Dataset</b><br>
We generated a 5x enlarged dataset of 19 JPG images and masks, each with 3500x3025 pixels, from the original STARE 700x605 pixels 
PPM.GZ image and label files using bicubic interpolation.
<br>
<br>
<b>2. Pre Augemtned Tiled STARE ImageMask Dataset</b><br>
We generated a pre-augmented image mask dataset from the enlarged dataset, which was tiledly-splitted to 512x512 pixels 
and reduced to 512x512 pixels image and mask dataset.
<br>
<br>
<b>3. Train Segmention Model </b><br>
We trained and validated a TensorFlow UNet model by using the <b>Pre Augmented Tiled STARE ImageMask Dataset</b>
<br>
<br>
<b>4. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict the STARE Retinal Vessel for the mini_test images 
with a resolution of 3500x3025 pixels of the Enlarged Dataset.<br><br>

<hr>
<b>Actual Tiled Image Segmentation for Images of 3500x3025 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0001.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0003.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0003.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0003.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0005.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this STARESegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been take from 
from the following images and labels
in <a href="https://cecas.clemson.edu/~ahoover/stare/">
STructured Analysis of the Retina
</a>
:<br><br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar">
<b>
Twenty images used for experiments
</b>
</a>
<br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar">
<b>
Hand labeled vessel network provided by Adam Hoover
</b>
</a>
<br>
<br>
Please see also <a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md">
STARE
</a>
<br>
<br>
<b>Authors and Institutions</b><br>
Adam Hoover (Department of Electrical and Computer Engineering, Clemson University)<br>
Valentina Kouznetsova (Vision Computing Lab, Department of Electrical and Computer Engineering, <br>
University of California, San Diego, La Jolla)<br>
Michael Goldbaum (Department of Ophthalmology, University of California, San Diego)
<br>
<br>
<b>Citation</b><br>
@ARTICLE{845178,<br>
  author={Hoover, A.D. and Kouznetsova, V. and Goldbaum, M.},<br>
  journal={IEEE Transactions on Medical Imaging}, <br>
  title={Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response}, <br>
  year={2000},<br>
  volume={19},<br>
  number={3},<br>
  pages={203-210},<br>
  doi={10.1109/42.845178}}<br>
<br>
<h3>
<a id="2">
2 Augmented-Tiled-STARE ImageMask Dataset
</a>
</h3>
 If you would like to train this STARE Segmentation model by yourself,
 please download the pre-augmented dataset from the google drive  
<a href="https://drive.google.com/file/d/18PvKykorWlO-dnZq9njCU0kVCa24w8Us/view?usp=sharing">
Augmented-Tiled-STARE-ImageMask-Dataset.zip</a>,
 expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Augmented-Tiled-STARE
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
This is a 512x512 pixels pre augmented tiles dataset generated from 3500x3025 pixels 19 <b>Enlarged-images</b> and
their corresponding <b>Enlarged-masks</b>.<br>
.<br>
We excluded all black (empty) masks and their corresponding images to generate our dataset from the original one.<br>  

The folder structure of the original stare-images and labels-ah data is the following.<br>

<pre>
./STARE
   ├─stare-images
   │  ├─im0001.ppm.gz
   │  ├─im0002.ppm.gz
   │  ├─...
   │  └─im0319.ppm.gz
   └─labels-ah
       ├─im0001.ah.ppm.gz
       ├─im0002.ah.ppm.gz
       ├─...
       └─im0319.ah.ppm.gz
</pre>
We excluded im0324.ah.ppm.gz file in the original labels-ah folder, because no corresponding 
im0324.ppm.gz was in stare-images folder.  <br>
On the derivation of this tiled dataset, please refer to the following Python scripts.<br>
<li><a href="./generator/Preprocessor.py">Preprocessor.py</a></li>
<li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_tiled_master.py">split_tiled_master.py</a></li>
<br>


<br>
<b>Augmented-Tiled-STARE Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/Augmented-Tiled-STARE_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough 
to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained STARE TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters=16</b> and large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Dataset class</b><br>
Specified ImageMaskDataset class.
<pre>
[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_LINEAR"
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 20
</pre>

<b>Tiled inference</b><br>
We used 3500x3025 pixels enlarged images and masks generated by <a href="./generator/Preprocessor.py">
Preprocessor.pys
</a>  as a mini_test dataset for our TiledInference.
<pre>
[tiledinfer] 
overlapping   = 64
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output_tiled"
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer      = False
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 6
</pre>

By using this callback, on every epoch_change, the epoch change tiledinfer procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/epoch_change_tiled_infer_at_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at ending (98,99,100)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/epoch_change_tiled_infer_at_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was terminated at epoch 100.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/train_console_output_at_epoch_100.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for STARE.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/evaluate_console_output_at_epoch_100.png" width="720" height="auto">
<br><br>Image-Segmentation-STARE

<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Augmented-Tiled-STARE/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.1236
dice_coef,0.8544
</pre>
<br>

<h3>
5 Tiled inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for STARE.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images (3500x3025 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled inferred test masks (3500x3025 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 3500x3025 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0002.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0004.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0044.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0077.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0077.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0077.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0081.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0081.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0081.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/images/im0139.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test/masks/im0139.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-STARE/mini_test_output_tiled/im0139.jpg" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Locating Blood Vessels in Retinal Images</b><br>
by Piecewise Threshold Probing of a<br>
Matched Filter Response<br>
Adam Hoover, Valentina Kouznetsova, and Michael Goldbaum<br>

<a href="https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf">
https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf
</a>
<br>
<br>
<b>2. STructured Analysis of the Retina</b><br>
<a href="https://cecas.clemson.edu/~ahoover/stare/">https://cecas.clemson.edu/~ahoover/stare/
</a>
<br>
<br>
<b>3. STARE</b><br>
<a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md">
https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md
</a>
<br>
<br>
<b>4. State-of-the-art retinal vessel segmentation with minimalistic models</b><br>
Adrian Galdran, André Anjos, José Dolz, Hadi Chakor, Hervé Lombaert & Ismail Ben Ayed<br>
<a href="https://www.nature.com/articles/s41598-022-09675-y">
https://www.nature.com/articles/s41598-022-09675-y
</a>
<br>
<br>
<b>5. Retinal blood vessel segmentation using a deep learning method based on modified U-NET model</b><br>
Sanjeewani, Arun Kumar Yadav, Mohd Akbar, Mohit Kumar, Divakar Yadav<br>
<a href="https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3">
https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3</a>
<br>
<br>

<b>6, Tensorflow-Image-Segmentation-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel</a>
<br>
<br>
<b>7. Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer
</a>
<br>
<br>
<b>8. Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma
</a>
<br>
<br>

<b>9. Tiled-ImageMask-Dataset-Breast-Cancer</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Breast-Cancer">
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Breast-Cancer
</a>
<br>
<br>

