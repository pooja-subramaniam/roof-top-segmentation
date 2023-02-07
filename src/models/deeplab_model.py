
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from typing import Any


def createDeepLabv3(out_channels: int=1) -> Any:
  """Creates DeepLabv3 model with pretrained weights and custom head
  Args:
      out_channels (int, optional): The number of output channels
      in your dataset labels. Defaults to 1.
  Returns:
      model: Returns the DeepLabv3 model with the ResNet101 
      backbone with pretrained weights and a classifier head.
  """
  model = models.segmentation.deeplabv3_resnet101(progress=True)
 
  # 2048 is the output channels from deeplabv3
  model.classifier = DeepLabHead(2048, out_channels)

  # Set the model in training mode
  model.train()
  return model