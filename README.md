# Diff_PaMIR
Adding diffusion model to Pamir.

## Demo

Please run the following commands to download necessary assets (including the pre-trained models):
```bash
cd ./networks
wget https://github.com/ZhengZerong/PaMIR/releases/download/v0.0/results.zip
unzip -o results.zip
cd ..
```

After that, run the following script to test the pre-trained network:
```bash
cd ./networks
python main_test.py
cd ..
```
This command will generate the textured reconstruction with the fitted SMPLs for the example input images in ```./network/results/test_data*/```. Note that we assume the input images are tightly cropped with the background removed and the height of the persons is about 80% of the image height (Please see the example input images we provide). 



## Dataset Generation for Network Training
In ```dataset_example```, we provide an example data item, which contains a textured mesh and a SMPL model fitted to the mesh. The mesh is downloaded from [RenderPeople](https://renderpeople.com/sample/free/rp_dennis_posed_004_OBJ.zip).  To generate the training images, please run:
```bash
cd ./data
python main_normalize_mesh.py                         # we normalize all scans into a unit bounding box
python main_calc_prt.py
python main_render_images.py
python main_sample_occ.py
python main_associate_points_with_smpl_vertices.py    # requires SMPL fitting
cd ..
```
Note that the last python script requires SMPL model fitted to the scans. To fit SMPL to your own 3D scans, you can use our tool released at [this link](https://github.com/ZhengZerong/MultiviewSMPLifyX). 

## Train the Network
Please run the following command to train the network:
```bash
cd ./networks
bash ./scripts/train_script_geo.sh  # geometry network
bash ./scripts/train_script_tex.sh  # texture network
cd ..
```
