#!/bin/bash

# create an empty screenshots directory
rm -rf ../screenshots
mkdir ../screenshots

# now, build and run the unity project
UNITY_PATH=/path/to/unity/editor/executable # Replace with the actual path to your Unity editor executable
PROJECT_PATH=/path/to/unity/project/folder   # Replace with the actual path to your Unity project folder

# Build the Unity project
$UNITY_PATH -projectPath $PROJECT_PATH -batchmode -quit -executeMethod YourBuildScript.Build

# Run the built game
$PROJECT_PATH/Builds/YourGameExecutable