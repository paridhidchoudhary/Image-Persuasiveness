Dataset:
https://drive.google.com/drive/folders/1i8fbGVC8C16eRIJ55ZrPQWE2lorp7qGI
1. final data
2. subset verification 
3. dataset_response
4. dataset_image


#### dataset_image are images for each group

inside there are 9 category names 
inside that there are 50 groups in each category 
inside that there images like 
original , edit-1 , edit-2 etc


#### the initial VLM annotations are saved inside dataset response 

inside there are 9 category names 
inside that there are 50 groups in each category 
inside that there responses like 
qwen-zeroshot , qwen-fewshot, pixtral-fewshot, pixtral-zeroshot etc

also we have user_output preference in user_output

#### subset verification 

groups for second level verification 

inside two folder -> images and response for 2nd level verification


#### final data 

it only have the response of the final dataset 
inside there are 9 category names 
inside that there are 50 groups in each category 
inside that there responses like 
user_output. 

#### actual data 
so the actual bench mark dataset images are in 
1. dataset_image
and the resposes are in 
1. final data

#### output format 

for each image we have :
{
description 
persuasion score 
}

ranking & reasoning 

conclusion 