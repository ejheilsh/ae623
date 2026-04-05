#!/bin/bash
# Script to clean up old data files

echo "Cleaning up old data files..."
rm -f data_steady/*.bin
rm -f data/results_*.bin
rm -f data/entropy_field_*.png
rm -f data/force_history.png
rm -f unsteady_plots/*
echo "Cleanup complete!"