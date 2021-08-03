#!/bin/bash
python Generation_training-deexperts.py Emotion joy True 0 
python Generation_training-deexperts.py Emotion joy false 0
wait
sleep 15s
python Generation_training-deexperts.py Emotion anger true 0 
python Generation_training-deexperts.py Emotion anger false 0
wait
sleep 15s
python Generation_training-deexperts.py Emotion fear true 0 &
python Generation_training-deexperts.py Emotion fear false 1
wait
sleep 15s
python Generation_training-deexperts.py Emotion love true 0 &
python Generation_training-deexperts.py Emotion love false 1
wait
sleep 15
python Generation_training-deexperts.py Emotion surprise true 0 &
python Generation_training-deexperts.py Emotion surprise false 1
wait
sleep 15
python Generation_training-deexperts.py Toxicity toxic true 0 &
python Generation_training-deexperts.py Toxicity toxic false 1 
wait
sleep 15
python Generation_training-deexperts.py Politeness polite true 0 &
python Generation_training-deexperts.py Politeness polite false 1 