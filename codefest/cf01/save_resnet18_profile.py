import torchvision.models as models
from torchinfo import summary

model = models.resnet18()

stats = summary(
    model,
    input_size=(1, 3, 224, 224),
    depth=3,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    verbose=0
)

with open("resnet18_profile.txt", "w", encoding="utf-8") as f:
    f.write(str(stats))

print("ResNet-18 profile saved successfully!")
