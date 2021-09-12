This is the code for the **SA-SOR** metric proposed in the paper. 

In order to run the code, please prepare the `input_data` following the rules below:

>1. The `input_data` is a list in which an element (`img_data`) is corresponding to an image.  
>2. The `img_data` is a dictionary containing the data required by **SA-SOR** metric. The data inculding `gt_mask`, `segmaps`, `gt_ranks`, and `rank_scores`.
>3. The 'gt_mask' is a list containing all mask of ground truth salient instance. The element has the shape of `h x w`, where `h` and `w` are the height and width of the testing image.
