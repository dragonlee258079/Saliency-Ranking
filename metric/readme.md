This is the code for the **SA-SOR** metric proposed in the paper. 

In order to run the code, please prepare the `input_data` following the rules below:

>1. The `input_data` is a list in which an element (`img_data`) corresponding to an image.  
>2. The `img_data` is a dictionary containing the data required by the **SA-SOR** metric, including `gt_mask`, `segmaps`, `gt_ranks`, and `rank_scores`.
>3. The `gt_mask` is a list containing ground truth masks of `m` salient instances. The elements in `gt_mask` have the shape of `h x w`, where `h` and `w` are the height and width of the testing image.
>4. The `segmaps` represent the prediction masks of salient instances. It is an array shaped as `n x h x w`, where `n` is the number of the predicted salient instances.
>5. The `gt_ranks` is a list containing `m` integers. Each integer represent the salient level for the corresponding salient instances. The higher integer stand the higher salient level.
>6. The `rank_scores` is a list containing `n` float number, representing the predicted salient level scroes for `n` prediction salient instances. The higher number stand the higher salient level. 
