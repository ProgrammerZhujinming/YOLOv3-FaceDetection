import torch
source_pre_model_pth = "./weights/YOLO_V3_130.pth"
destination_model_pth = "./weights/YOLO_V3_final.pth"
param_dict = torch.load(source_pre_model_pth, map_location=torch.device("cpu"))['model']
torch.save(param_dict, destination_model_pth)