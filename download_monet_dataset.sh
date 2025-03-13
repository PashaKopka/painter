#!/bin/bash

# Download the monet dataset
kaggle competitions download -c gan-getting-started

# Create a directory for the dataset
mkdir data

# Unzip the dataset
unzip -q gan-getting-started.zip -d ./data