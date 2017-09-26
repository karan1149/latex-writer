# LaTeX Writer

The goal of this project is to learn more about and experiment with generative adversarial networks, an exciting new technique in machine learning. GANs are cool because they generate images that appear similar to training data, but are actually entirely new images. For example, if you train a GAN with images of dogs, after training the GAN will be able to generate completely new images of dogs! In a sense, the the neural network appears to be learning about what a dog is, more so than if it could just classify different breeds of dogs. This is a step toward machines that really appear to be intelligent!

## GANs

In brief, GANs work by training two networks: the generator G and the discriminator D. The generator generates images, and the discrimator is a binary classifier that determines whether a given image came from either the generator or the training data. The goal of the generator is to generate images that fool the discriminator into classifying them as real images. Initially, neither model is particularly good at its job, but the models are trained against each other so the discriminator gets better at telling generated images apart from training data while the generator gets better at fooling the discriminator. See [this article](https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7) for a visual explanation.

## AC-GANs

There have been many advances in GANs since their inception in 2014, and one of the more interesting (albeit simple) variants is [Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585). AC-GANs modify the generator so that it takes in the identity of a class of image to produce (a certain breed of dog, for instance) and modify the discriminator so that it also predicts the class given an image. The end result is a model that can produce new images of a desired class, as well as classify images. 

I implemented a simplified version of AC-GANs in Keras (based on Luke de Oliveira's GAN code). This works well with the MNIST dataset, generating plausible handwritten digits after 13 epochs:

![MNIST generated digits](https://github.com/karan1149/latex-writer/raw/master/model_outputs/mnist_digits_generated.png)

## Dataset

After implementing AC-GANs, I wanted to test with a different dataset, since the MNIST is a relatively simple dataset that often does not tell you much about whether a model will work on more complex data. That said, I do not have access to the computing power that I would need to train on and generate high resolution data (and generating high resolution output is still a work in progress with GANs), so I generated a dataset that fit my needs by building an [extractor](https://github.com/karan1149/crohme-data-extractor) for the CROHME dataset of handwritten mathematical expressions. 

The important properties of this dataset (for experimenting with the robustness of AC-GANs) are that there is much more variation between images of a given class in this data, the data is limited (I deliberately capped the data at 35,840 examples in `image_utils.py`), and the number of images per class varies widely (from 930 to 2909). The last property is especially interesting for my purposes, because the typical training procedure for AC-GANs involves sampling from a probability distribution of classes, which is typically assumed to be a uniform distribution. Other than these points, the extracted data generally resembles the MNIST data (28x28, monochrome, handwritten).

## Experiments

First, I ran the AC-GAN implementation as is on the new dataset. I had to change the Adam learning rate to 0.00005 from 0.0001 because model weights diverged otherwise. The results were significantly worse than on the MNIST dataset. After 200 epochs (around which the model errors held roughly constant), we have the following plot:

![CHROHME unoptimized plot with equal LR](https://github.com/karan1149/latex-writer/raw/master/model_outputs/outputs_uniform_1/plot_epoch_199_generated.png)

Each of 16 columns in this plot corresponds to a different symbol class. In order, these symbols are: "(", ")", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "x". For each column, the model makes 10 attempts to generate the corresponding symbol. We can see that for this dataset, the model does relatively poorly. For most symbols, it has an idea of what they are supposed to look like, but for symbols without much training data, the generated images are pretty bad ("6", "7", "8", "9" are all examples, as they all have <1000 training data examples, whereas all the other classes have >2000).

Looking at the training logs for this run, I noticed that the generator loss was relatively high compared to the discriminator loss. Especially toward the beginning of training, it appeared that the discriminator was getting better, but the training step that modified the generator was working slower, and the generator was getting outpaced. This was true even though my implementation of AC-GAN training used the same number of examples (2 * batch size) to train both models at each step. This is an important consideration for the robustness of the model because if the generator does not get better fast enough, then the discriminator (which is competing with the generator) will not get better fast enough.

To fix this, I tried using a different learning rate for the generator and the discriminator (generator has 1.5x learning rate), which gave significantly improved results after the same number of epochs:

![CHROHME unoptimized plot](https://github.com/karan1149/latex-writer/raw/master/model_outputs/outputs_uniform_1.5/plot_epoch_199_generated.png)

Notice that the results are much better for those less frequent symbols. Also notice the variation within class for some of the generated images: some of the "4"s and "7"s are written differently.

I also tried setting the generator to have 2x the learning rate of the discriminator, but the results seem to be about the same:

![CHROHME unoptimized plot with 2x LR](https://github.com/karan1149/latex-writer/raw/master/model_outputs/outputs_uniform_2/plot_epoch_199_generated.png)

Another experiment I did relates to the aforementioned probability distribution of classes. Since some classes in my dataset contained 3x the data of other classes, I changed the class probability distribution used for training to weight classes based on the amount of training data they have. This ensures that data of each class is represented equally during training. 

Using a generator learning rate 1.5x greater than the discriminator learning rate, the results I got using this method are shown in this plot of results after 100 epochs:

![CHROHME optimized plot](https://github.com/karan1149/latex-writer/raw/master/model_outputs/outputs_weighted_1.5/plot_epoch_199_generated.png)


As you would expect, the model now does better in generating images for classes that are poorly represented in the training data ("6", "7", "8", "9"). However, this appears to be at the cost of performance on other symbols, most clearly "0" in this case. The results are worse if the generator learning rate is 2x greater than the discriminator learning rate:

![CHROHME optimized plot with 2x LR](https://github.com/karan1149/latex-writer/raw/master/model_outputs/outputs_weighted_2/plot_epoch_199_generated.png)

In sum, changing the class probability distribution appears to have mixed results. 

## Conclusion

These small experiments were a great way to learn more about GANs, CNNs, Keras, and deploying models on Google Cloud Platform. There is still plenty of work to be done on improving the robustness and stability of GAN models. Authors have generally reported results only for low resolution (usually <= 128x128) images on relatively simple datasets, and a relatively small change in dataset can completely break training, as I saw initially when I first replaced the MNIST dataset with the CROHME one. [This paper](https://arxiv.org/abs/1606.03498) shows some exciting efforts in this direction (co-authored by Ian Goodfellow, who invented GANs), and many more are sure to come.

