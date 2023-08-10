# Replication of "[Open Problems in Computer Vision for Wilderness SAR and The Search for Patricia Wu-Murad](https://arxiv.org/abs/2307.14527)"

1) Checkout the [ai-hadr-iccv-2023](https://github.com/TManzini/WiSAR/tree/ai-hadr-iccv-2023) branch of this repository.
2) Download the original imagery from the search [here]().
3) Download the models referenced in the paper: [epoch=174-step=25725.ckpt](https://www.dropbox.com/scl/fi/0qj9g6ojl27eh60phnp2i/epoch-174-step-25725.ckpt?rlkey=qmfexrft0lhhkn32r4sscod7j&dl=0), [epoch=84-step=33490.ckpt](https://www.dropbox.com/scl/fi/0y7wp7ravkb310jxdguyz/epoch-84-step-33490.ckpt?rlkey=j8n6y1czt79nwzx008icb1ivg&dl=0).

## To generate the frames that are to be fed to the different detection systems
1) Extract frames from the videos that you want to analyze, and copy the raw images.
    
    Since these models only consume images, instead of videos, you need to extract frames from the videos that we will run the models on. Specifically, you will need to generate frames of the video you want to analyze, and a mapping between the extracted frames and the original videos (so you can look up the video from the extracted imagery).

    This can be done by calling `folder2frames.py` file in `WiSAR/src/utils`. Here is an example call...
    
    ```bash
    cd WiSAR/src/utils/
    python folder2frames.py --in_folder_path <PATH_TO_DOWNLOADED_WU_MURAD_DATASET> --out_id2path_path /WiSAR/out/id2path.csv --out_folder_path /WiSAR/out/extracted_frames --fps_sample_rate 2
    #Note: There are further parameters that are specified in the script.
    #Note: --fps_sample_rate indicates the number of frames per second that should be sampled.
    ```

## To Replicate the results of the RX classifier on the Wu-Murad Dataset...

1) Download the images from the Wu-Murad Dataset [here]()
2) Run Inference on these images by calling `RXInference.py` located at `WiSAR/src/inference_scripts/` as follows
    ```bash
    cd WiSAR/src/inference_scripts/
    python RXInference.py --in_folder_path /WiSAR/out/extracted_frames --out_folder_path /WiSAR/out/ --id2path_file_path /WiSAR/out/id2path.csv
    ```

## To Replicate the results of the Tiled_EffecientDET<sub>84</sub> and Tiled_EffecientDET<sub>174</sub> models on the Wu-Murad Dataset...

1) Download the images from the Wu-Murad Dataset [here]()
2) Run Inference on these images by calling `EfficientDetTiledInference.py` located at `WiSAR/src/inference_scripts/` as follows
    ```bash
    cd WiSAR/src/inference_scripts/
    python EfficientDetTiledInference.py --in_folder_path /WiSAR/out/extracted_frames --model_path /WiSAR/models/EfficientDet/HERIDAL/epoch=84-step=33490.ckpt --out_folder_path /WiSAR/out/ --id2path_file_path /WiSAR/out/id2path.csv
    python EfficientDetTiledInference.py --in_folder_path /WiSAR/out/extracted_frames --model_path /WiSAR/models/EfficientDet/HERIDAL/epoch=174-step=25725.ckpt --out_folder_path /WiSAR/out/ --id2path_file_path /WiSAR/out/id2path.csv
    #Note: If you encounter cuda out of memory errors, consider reducing the batch size using the --batch_size argument.
    ```

## To Replicate the results on the HERIDAL Dataset...
1) Download the HERIDAL dataset from the publishing authors [here](http://ipsar.fesb.unist.hr/HERIDAL%20database.html).
2) Convert the labels from the HERIDAL Dataset from XML to CSV using the `convert_heridal_xmls_to_csv.py` file in `WiSAR/src/datasets/HERIDAL/` as follows.
    ```bash
    cd WiSAR/src/datasets/HERIDAL/
    python convert_heridal_xmls_to_csv.py --image_folder_path <PATH_TO_HERIDAL_IMAGE_TEST_FOLDER> --label_folder_path <PATH_TO_HERIDAL_LABELS_TEST_FOLDER> --out_csv_path <PATH_TO_OUTPUT_CSV> 
    ```
3) To replicate the results on the HERIDAL dataset using the Tiled_EffecientDET<sub>84</sub> model, perform inference on the HERIDAL dataset using the `EvaluateHERIDALTestSet.py` file located in `WiSAR/src/inference_scripts/` as follows.
    ```bash
    cd WiSAR/src/inference_scripts/
    python EvaluateHERIDALTestSet.py --in_folder_path <PATH_TO_HERIDAL_IMAGE_TEST_FOLDER> --in_labels_file_path <PATH_TO_HERIDAL_LABELS_CSV> --out_folder_path <PATH_TO_OUTPUT_PREDS_CSV> --model_path ../../models/EfficientDet/HERIDAL/epoch=84-step=33490.ckpt --tile_dim 512 --model_confidence_threshold 0.0 --union_overlapping_bboxes
    #Note: If you encounter cuda out of memory errors, consider reducing the batch size using the --batch_size argument.
    ```
4) To replicate the results on the HERIDAL dataset using the Tiled_EffecientDET<sub>174</sub> model, perform inference on the HERIDAL dataset using the `EvaluateHERIDALTestSet.py` file located in `WiSAR/src/inference_scripts/` as follows.
    ```bash
    cd WiSAR/src/inference_scripts/
    python EvaluateHERIDALTestSet.py --in_folder_path <PATH_TO_HERIDAL_IMAGE_TEST_FOLDER> --in_labels_file_path <PATH_TO_HERIDAL_LABELS_CSV> --out_folder_path <PATH_TO_OUTPUT_PREDS_CSV> --model_path ../../models/EfficientDet/HERIDAL/epoch=174-step=25725.ckpt --tile_dim 512 --model_confidence_threshold 0.0 --union_overlapping_bboxes
    #Note: If you encounter cuda out of memory errors, consider reducing the batch size using the --batch_size argument.
    ```
5) With the predictions csv that was generated in either steps 4 or 5 above you can then compute the performance metrics using the `compute_bounding_box_metrics.py` file located in `WiSAR/src/inference_scripts/` as follows.
    ```bash
    cd WiSAR/src/inference_scripts/
    python compute_bounding_box_metrics.py --in_labels_file_path <PATH_TO_HERIDAL_LABELS_CSV> --in_preds_file_path <PATH_TO_HERIDAL_PREDS_CSV>
    ```
