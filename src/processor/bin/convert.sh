#!/bin/bash
ffmpeg -i $1.m4a -ar 16000 -map_channel 0.0.0 $1.wav
ls
