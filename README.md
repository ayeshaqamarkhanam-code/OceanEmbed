# Ocean Subsurface Prediction Model - SIH 2026 
 
## Project Structure 
```text 
SIH 2026/ 
ÃÄÄ INCOIS_X_surface.npy         # Input Data (Surface Variables) 
ÃÄÄ INCOIS_Y_subsurface.npy      # Target Data (Subsurface Depth Levels) 
ÃÄÄ data_loader.py               # Dataset processing and PyTorch DataLoader 
ÃÄÄ model.py                     # SatelliteEncoder, SubsurfaceDecoder, Autoencoder 
ÀÄÄ test_integration.py          # Integration verification script 
``` 
 
## Dataset Setup 
The `.npy` dataset files are not tracked in this repository due to file size limits. 
 
Please download `INCOIS_X_surface.npy` and `INCOIS_Y_subsurface.npy` from the shared drive and place them in the root directory. 
 
## Integration Verification 
To test that the dataset pipeline and neural network architecture integrate seamlessly, run: 
```bash 
python test_integration.py 
``` 
