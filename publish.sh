#! /bin/bash
export LABROOT=$HOME/labs

# Store the source directory path
SOURCE_DIR=$PWD

# Go to destination and clean it
cd $LABROOT/sketchy_moment_matching

# remove all files in the directory except .git
find . -mindepth 1 -not -path './.git*' -delete

# Use rsync to copy files while excluding .git directory
rsync -avzrP --exclude='.git' \
    $SOURCE_DIR/ \
    $LABROOT/sketchy_moment_matching/

# Commit and push changes
git add .
git commit -m "publish"
git push
