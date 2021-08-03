#!/bin/bash
#python Generation_training-deexperts.py Emotion joy True 0 
python Generation_training-deexperts.py Emotion joy false 0
wait
sleep 15s
python Generation_training-deexperts.py Emotion anger true 0 
python Generation_training-deexperts.py Emotion anger false 0
wait
sleep 15s
