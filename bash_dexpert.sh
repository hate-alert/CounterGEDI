#!/bin/bash
# python Generation_training-deexperts.py Toxicity toxic 0 &                                                                       
# python Generation_training-deexperts.py Toxicity non-toxic 1  
# wait
# sleep 15s
# python Generation_training-deexperts.py Humour non-humor 0 &                                                                       
# python Generation_training-deexperts.py Humour humor  1 
# wait
# sleep 15s
# python Generation_training-deexperts.py Politeness polite  0 &
# python Generation_training-deexperts.py Politeness polite  1 
python Generation_training-deexperts.py Emotion joy  0 &
python Generation_training-deexperts.py Emotion sadness  0 
wait
sleep 15s
python Generation_training-deexperts.py Emotion anger  0 &
python Generation_training-deexperts.py Emotion fear  0 
wait
sleep 15s
python Generation_training-deexperts.py Emotion love  0 &
python Generation_training-deexperts.py Emotion surprise 0
wait
sleep 15s
python Generation_training-deexperts.py Toxicity non_toxic 0 &
python Generation_training-deexperts.py Politeness non_polite  0 
