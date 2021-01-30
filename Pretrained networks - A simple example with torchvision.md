# Pretrained networks - A simple example with torchvision

[Torchvision](https://pytorch.org/docs/stable/torchvision/index.html) is part of the [Torch](https://pytorch.org/) project. Torchvision includes pretrained models for vision tasks.

the module **models** contains many vision models.

```python
# models contains pretrained models
from torchvision import models

# Instantiating resnet
# If pretrained is set to False, the weights of the pretrained model
# are not downloaded, and so the model has yet to be trained.
resnet = models.resnet101(pretrained=True)
```

The pretrained resnet101 model included in pytorch was trained on Imagenet. Imagenet includes 1000 categories. The pretrained network spits out 1000 outputs, with a prediction for each class. This is going to come in play once we get our predictions.

The pretrained resnet model needs the images to have a very specific size. One has to resize the images before feeding them to the model. Applying the **same transformations** as the ones applied while training the model is mandatory.

`````python
# transforms includes common image transformations
from torchvision import transforms

# We chain the transformations using transforms.Compose
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        # The mean and std are those of the training data
        # We have to normalize our test data using the same mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
`````

Now that we have the model and our transformations, we can load an image and apply the transformations and get the model prediction

````python
# PIL is the Python Imaging Library
from PIL import Image

# We open an image of our choice using PIL
img = Image.open("any_image.jpg")

# We apply the transformations to our image object
img_transformed = preprocess(img) # Tensor object
````

Pytorch expects the first dimension of the the tensor that is fed to the neural net to be that of the batch size. Our image has a dimension of (3=Number of RGB colors, 224, 224). The 224 comes from our cropping transformations. We need to prepend the dimension of the batch. As we're feeding one image per time (just to test our network), we prepend a dimension of 1.

````python
import torch

# We prepend a dimension of 1, corresponding to the size of our batch
img_batch_transformed = torch.unsqueeze(img_transformed, 0) #=> New dimension: [1, 3, 224, 224]
````

We need to put the network in evaluation mode. This tells the network that some layers (like dropout) should behave in a different way than in training mode.

````python
# Evaluation mode
resnet.eval()

# We feed our image to the network to get the prediction
prediction = resnet(img_batch_transformed)
````

As said at the beginning, prediction will be of size [1, **1000**], a prediction for each one of the 1000 categories in Imagenet dataset.

To normalize the output to a range of [0, 1] and to get a sum of 1 for all predictions, we apply a softmax.

````python
# The one correspond to the dimension over which we want to normalize
# as we want to normalize over the 1000 prediction, we choose
# the second dimension (0=first dim, 1=second dim)
prob_prediction = torch.nn.functional.softmax(prediction, 1) # [1, 1000]
````

**Please,** **if you get lost when using pytorch tensors**, feel free to check the shape of the tensors along the way `my_tensor.shape`. It's a very good way of understanding what going on and what each operation does.

Now that we have prediction that we can interpret as probabilities, we'll find out the max value and its index

````python
value, index = torch.max(prob_prediction, 1)
````

value and index are not scalar, each one of them is put inside a pytorch tensor. To get just the scalar, you have to use the `item()` method.

````python
value.item() # The maximum prediction value - It can be interpreted as a probability
index.item() # The index of the class with the maximum prediction
````

