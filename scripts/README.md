This is the code pipeline for generating persuasion data based on images and then to finetune a model based on that so that we can identify the most persuasive image from a set of images based on some specific category product.

The code pipeline goes like this :

we start of with vismin -> https://arxiv.org/pdf/2407.16772
visual minimal changes ->  in this paper , they have proposed a dataset consisting of images with minimal changes and  they trained a vlm to learn these small changes which is very hard to do . 
Now our objective is to understand how small changes in the images changes the persuasiveness of the image with respect to some specific product . 


For this perticular purpose we have aalready selected 12 category. 
these are electronics : 1. Laptop, 2. Tv , 3. Cell phone, 4. remote
          kitchen appliances : 1. microwave , 2. toaster , 3. refrigerator 
          Luggage: 1. Suitcase , 2. Handbag , 3. Backpack
          Furniture: 1. Chair, 2. Couch



code : {### subgroup_dataset_creation.py} 

Now our major idea is to group the vis-min dataset based on these category 

setp 1. first we grouped images based on the original image and the edited images. means all edits done on the same original image is grouped in one group. 

step 2. we have grouped them based on the caption of the original image , if the image caption have the product name in it we have categorized that group in that product . 
obviously , it means that one group can be in multiple products . 

code : {### vismin-dataset-org.py}

Now, the major idea is that we have groups and category they belongs to . 

step 3. we have taken 50 samples from each category , to create a bench mark dataset. 

code: {### Dataset_create_human_verification.py}

step 4.
Idea is to get the output from different models in different settings and then get human feedback on that dataset . 

so we took 2 models -> 1. Pixtral - Large , 2. Qwen-2.5-7B-VL 
and we used two setting -> 1. zero shot , 2. one shot for both the models 

so for each dataset we got about 4 outputs 

and now we have got human feedback on this dataset , 
implying which of the 4 output is more accurate and also if none of them are correct then we took a human output for future usage . 

code : {### qwen_zeroshot.py , ### qwen-fewshot.py , ### pixtral_few_shot.py, ### pixtral_zeroshot.py}

step 5.
now for validation we will see the average agreement level between human user response(As one of the model response is choosen by the user) and the other 3 models . 
Now our idea is , to choose the examples with less than equal to 50% average agreement (with respect to the other 3 responses) and do revaluation on these image groups by two other indepdent usres to mitigate the bias . 

code : { ###user_agreement_level.py , ###dataset_for_second_human_eval.py}

step 6. 
we are now finetuning the model based on the best output based on human preference and then we are using different matrix to 
quantify the model accuracy. 



use general seed value 50 for this 

Training:
Various Iterations have been done:
1. Listwise : paridhi_model_training_listwise.py
2. Pairwise: paridhi_model_training_pairwise.py
3. 2 stage: Stage 1: LM loss: train_stage1_lm.py
            Stage 2: ranking loss: train_stage2_ranking.py

4. Score head only(lora + mean+max pooling): model_training_scorehead_only_lora.py
5. Score head only (lora + attention pooling): model_training_scorehead_only_lora_attention.py
6. Score head (Phase 1) + LM loss (phase 2): phase 1: model_training_scorehead_only_lora.py
                                            phase 2: model_training_lm_loss_phase2.py



Evaluation File:
1. Listwise: paridhi_model_eval_listwise.py
2. 2 stage: eval_stage2.py
3. score head + justifications: model_eval_scorehead_and_justifications.py
4. score head (attention pooling+ justifications): model_eval_scorehead_attnpool_with explanations.py

Final Configuration:
1. Score head(lora + mean+max pooling): model_training_scorehead_only_lora.py
2. Eval (with justifications): model_eval_scorehead_and_justifications.py


Experiments (Statistical analysis and other plots -> based on final configuration):
1. statistical analysis (boxplots): statistical_boxplots.py, statistical_tests_model_scorehead.py
2. experiment2_final_crossvalidation.py
3. experiment3_final_Score_distribution.py
4. experiment4_final_group_size.py

Baselines Implementation:
1. NIMA: NIMA_score.py, NIMA_eval.py
2. MUSIQ: go to MUSIQ directory-> musiq_eval.py, musiq_score.py


 

we have done the following code:
{qwen_zeroshot_tournament.py,qwen_fewshot_tournament.py,pixtral_zeroshot_tournament.py,pixtral_fewshot_tournament.py,llama_zeroshot_tournament.py}

evalution matrix -> 
1. MSE -> mean square error of persuasion score
2. Accuracy -> top rank accuracy 
3. Average agreement -> in ranking how much percentage of rank is same . 
4. Kendall's Tau -> Pairwise Ranking
5. Spearman's Rho -> overall ranking


other base lines:

NIMA.py(neural image assessment) and MUSIQ

for NIMA github link :
https://github.com/titu1994/neural-image-assessment
clone this and directly use 

For CAP use this paper for reference :
https://arxiv.org/abs/2412.10426

conda venv -> qwen-finetune for all qwen related codes
need to install mistral.ai and grok for pixtral and llama related codes



additional training 

now we want to do double training 

1st finetune a model using 
{### model-training-all-fulltext.py}

then get outputs for all groups using this model 
{finetuned_model_output.py}

then merge this model's adapter with the original base model qwen to do a second level finetuning 
{merge.py}

then do a second level finetuning , but this time -> provide the output that we got from this finetuned model as a input .
{two-stage-finetuning.py}

for inference and eval 
first get output from the initial finetunde model

then use this to get the output from the second level finetuned model 

evaluation code :
{M2-eval.py}

#### in case of any error in training codes 
1. try to change auto map to singular gpu core . like may be gpu 0
2. try to change evaluation_strategy to eval_strategy 


all results has been shared in this google doc :
https://docs.google.com/document/d/1nv1opR_9t-JJLIApMFP1YQplVdKXfU2q/edit?usp=sharing&ouid=102506909968642931076&rtpof=true&sd=true

all dataset has been share in :
https://drive.google.com/drive/folders/1i8fbGVC8C16eRIJ55ZrPQWE2lorp7qGI?usp=drive_link

please make a clone for all this 







