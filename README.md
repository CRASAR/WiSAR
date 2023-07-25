# WiSAR - (Wi)lderness (S)earch (A)nd (R)escue 
This Repository contains supervised and unsupervised methods for wilderness search and rescue.


    
## So, you recieved a dataset from a real world Wilderness SAR team...
1) Extract frames from the videos that you want to analyze, and copy the raw images.
    
    Since these models only consume images, instead of videos, you need to extract frames from the videos that we will run the models on. Specifically, you will need to generate frames of the video you want to analyze, and a mapping between the extracted frames and the original videos (so you can look up the video from the extracted imagery).

    This can be done by calling `folder2frames.py` file in `WiSAR/src/utils`. Here is an example call...
    
    ```bash
    cd WiSAR/src/utils/
    python folder2frames.py --in_folder_path <PATH_TO_RAW_DATA> --out_id2path_path /WiSAR/out/id2path.csv --out_folder_path /WiSAR/out/extracted_frames --fps_sample_rate 2
    #Note: There are further parameters that are specified in the script.
    #Note: --fps_sample_rate indicates the number of frames per second that should be sampled.
    ```

2) You then need to run models on this extracted imagery

    2.1) **RX Spectal Classifer** - This can be done by calling `RXInference.py` located at `WiSAR/src/inference_scripts/`. Here is an example call...

    ```bash
    cd WiSAR/src/inference_scripts/
    python RXInference.py --in_folder_path /WiSAR/out/extracted_frames --id2path_file_path /WiSAR/out/id2path.csv --out_folder_path /WiSAR/out/ 
    #Note: There are further parameters that are specified in the script.
    #Note: --id2path_file_path can be omitted, if step 1 was skipped.
    ```

    2.2) **EfficientDet Bounding Box Model** - You will need a model in order to run inference. The models of this type are located at `/WiSAR/models/`. Inference can be done by calling `EfficientDetTiledInference.py` located at `WiSAR/src/inference_scripts/`. Here is an example call to run one of these models...
        
    ```bash
    cd WiSAR/src/inference_scripts/
    python EfficientDetTiledInference.py --in_folder_path /WiSAR/out/extracted_frames --model_path /WiSAR/models/EfficientDet/HERIDAL/epoch=174-step=25725.ckpt --id2path_file_path /WiSAR/out/id2path.csv --out_folder_path /WiSAR/out/ 
    #Note: There are further parameters that are specified in the script.
    #Note: --id2path_file_path can be omitted, if step 1 was skipped.
    #Note: If you encounter cuda out of memory errors, consider reducing the batch size using the --batch_size argument.
    ```
   
3) Now you need to inspect the regions of interest that have been proposed by the models. The outputs should be in `WiSAR/out` (or wherever you specified). Happy squinting!
