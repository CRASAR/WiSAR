# Replication of "Open Problems in Computer Vision for Wilderness SAR and The Search for Patricia Wu-Murad"

1) Checkout the [ai-hadr-iccv-2023](https://github.tamu.edu/hrail/WiSAR/tree/ai-hadr-iccv-2023) branch of this repository.

## To Replicate the results of the RX classifier on the Wu-Murad Dataset...

1) Download the images from the Wu-Murad Dataset [here]()
2) Run Inference on these images by calling `RXInference.py` located at `WiSAR/src/inference_scripts/` as follows
    ```bash
    cd WiSAR/src/inference_scripts/
    python RXInference.py --in_folder_path <PATH_TO_DOWNLOADED_WU_MURAD_DATASET> --out_folder_path /WiSAR/out/ 
    ```

## To Replicate the results of the Tiled_EffecientDET<sub>84</sub> and Tiled_EffecientDET<sub>174</sub> models on the Wu-Murad Dataset...

1) Download the images from the Wu-Murad Dataset [here]()
2) Run Inference on these images by calling `EfficientDetTiledInference.py` located at `WiSAR/src/inference_scripts/` as follows
    ```bash
    cd WiSAR/src/inference_scripts/
    python EfficientDetTiledInference.py --in_folder_path <PATH_TO_DOWNLOADED_WU_MURAD_DATASET> --model_path /WiSAR/models/EfficientDet/HERIDAL/epoch=84-step=33490.ckpt --out_folder_path /WiSAR/out/ 
    python EfficientDetTiledInference.py --in_folder_path <PATH_TO_DOWNLOADED_WU_MURAD_DATASET> --model_path /WiSAR/models/EfficientDet/HERIDAL/epoch=174-step=25725.ckpt --out_folder_path /WiSAR/out/
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
