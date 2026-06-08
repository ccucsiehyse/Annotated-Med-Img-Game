# Annotated-Med-Img-Game
This is a game that allows you to annotate medical images and see the results of your annotations.

## Environment Setup
※ file structure see ```file_tree.txt```
1. Install the dependencies
2. git clone MedSAM (by bowang-lab)
3. download model checkpoint "medsam_vit_b.pth" (by bowang-lab (github))
4. download dataset from kaggle "Brain CT Images with Intracranial Hemorrhage Masks" (by vbookshelf)
5. go Google AI Studio and get google gemini api key
6. paste the google gemini api key into the med_game.py file

## Run the game
```bash
python med_game.py
```

## Gameplay
1. the game will show you a medical image
2. you can annotate the image by drawing a bounding box with mouse
3. press "H" to toggle the hint of the image
4. press "R" to clear the bounding box
5. press "Enter" to submit the annotation
6. the game will show you the result of your annotations
7. the game will give you a score based on the accuracy of your annotations
8. the game will give you a feedback based on the accuracy of your annotations