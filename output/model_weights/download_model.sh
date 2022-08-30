#!/bin/bash

cd output/model_weights

echo download full model
wget https://www.dropbox.com/s/uc10gl58h6t6c62/full_model.zip?dl=1
unzip full_model.zip
rm full_model.zip

echo download model with resnet ablation
wget https://www.dropbox.com/s/iwzp94zsj0g8syq/minus_resnet.zip?dl=1
unzip minus_resnet.zip
rm minus_resnet.zip

echo download model with layoutlm ablation
wget https://www.dropbox.com/s/6f2akulr4labhtz/minus_layoutlm.zip?dl=1
unzip minus_layoutlm.zip
rm minus_layoutlm.zip
