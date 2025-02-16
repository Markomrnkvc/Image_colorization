# Image_colorization
![](https://github.com/Markomrnkvc/Image_colorization/blob/readme/gif_colorization.gif)
---------
#### Install with Anaconda

You should now be able to do a simple install with Anaconda. Here are the steps:

Open the command line and navigate to the folder where you want to download the repo.  Then
type the following commands

```console
git clone https://github.com/Markomrnkvc/Image_colorization.git
cd DeOldify
conda env create -f environment.yml
```

Then you can start using the code! 


From there you can start running the notebooks in Jupyter Lab, via the url they
provide you in the console.

> **Note:** You can also now do "conda activate deoldify" if you have the latest
version of conda and in fact that's now recommended. But a lot of people don't
have that yet so I'm not going to make it the default instruction here yet.

**Alternative Install:** User daddyparodz has kindly created an installer script
for Ubuntu, and in particular Ubuntu on WSL, that may make things easier:
  <https://github.com/daddyparodz/AutoDeOldifyLocal>

#### Note on test_images Folder

The images in the `test_images` folder have been removed because they were using
Git LFS and that costs a lot of money when GitHub actually charges for bandwidth
on a popular open source project (they had a billing bug for while that was
recently fixed).  The notebooks that use them (the image test ones) still point
to images in that directory that I (Jason) have personally and I'd like to keep
it that way because, after all, I'm by far the primary and most active developer.
But they won't work for you.  Still, those notebooks are a convenient template
for making your own tests if you're so inclined.

#### Typical training

The notebook `ColorizeTrainingWandb` has been created to log and monitor results
through [Weights & Biases](https://www.wandb.com/). You can find a description of
typical training by consulting [W&B Report](https://app.wandb.ai/borisd13/DeOldify/reports?view=borisd13%2FDeOldify).

## Pretrained Weights

To start right away on your own machine with your own images or videos without
training the models yourself, you'll need to download the "Completed Generator
Weights" listed below and drop them in the /models/ folder.

The colorization inference notebooks should be able to guide you from here. The
notebooks to use are named ImageColorizerArtistic.ipynb,
ImageColorizerStable.ipynb, and VideoColorizer.ipynb.

### Completed Generator Weights

- [Artistic](https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth)
- [Stable](https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=0)
- [Video](https://data.deepai.org/deoldify/ColorizeVideo_gen.pth)

### Completed Critic Weights

- [Artistic](https://www.dropbox.com/s/xpq2ip9occuzgen/ColorizeArtistic_crit.pth?dl=0)
- [Stable](https://www.dropbox.com/s/s53699e9n84q6sp/ColorizeStable_crit.pth?dl=0)
- [Video](https://www.dropbox.com/s/xnq1z1oppvgpgtn/ColorizeVideo_crit.pth?dl=0)

### Pretrain Only Generator Weights

- [Artistic](https://www.dropbox.com/s/h782d1zar3vdblw/ColorizeArtistic_PretrainOnly_gen.pth?dl=0)
- [Stable](https://www.dropbox.com/s/mz5n9hiq6hmwjq7/ColorizeStable_PretrainOnly_gen.pth?dl=0)
- [Video](https://www.dropbox.com/s/ix993ci6ve7crlk/ColorizeVideo_PretrainOnly_gen.pth?dl=0)

### Pretrain Only Critic Weights

- [Artistic](https://www.dropbox.com/s/gr81b3pkidwlrc7/ColorizeArtistic_PretrainOnly_crit.pth?dl=0)
- [Stable](https://www.dropbox.com/s/007qj0kkkxt5gb4/ColorizeStable_PretrainOnly_crit.pth?dl=0)
- [Video](https://www.dropbox.com/s/wafc1uogyjuy4zq/ColorizeVideo_PretrainOnly_crit.pth?dl=0)

## Want the Old DeOldify?

We suspect some of you are going to want access to the original DeOldify model
for various reasons.  We have that archived here:  <https://github.com/dana-kelley/DeOldify>

## Want More?

Follow [#DeOldify](https://twitter.com/search?q=%23Deoldify) on Twitter.

## License

All code in this repository is under the MIT license as specified by the LICENSE
file.

The model weights listed in this readme under the "Pretrained Weights" section
are trained by ourselves and are released under the MIT license.

## A Statement on Open Source Support

We believe that open source has done a lot of good for the world.  After all,
DeOldify simply wouldn't exist without it. But we also believe that there needs
to be boundaries on just how much is reasonable to be expected from an open
source project maintained by just two developers.

Our stance is that we're providing the code and documentation on research that
we believe is beneficial to the world.  What we have provided are novel takes
on colorization, GANs, and video that are hopefully somewhat friendly for
developers and researchers to learn from and adopt. This is the culmination of
well over a year of continuous work, free for you. What wasn't free was
shouldered by us, the developers.  We left our jobs, bought expensive GPUs, and
had huge electric bills as a result of dedicating ourselves to this.

What we haven't provided here is a ready to use free "product" or "app", and we
don't ever intend on providing that.  It's going to remain a Linux based project
without Windows support, coded in Python, and requiring people to have some extra
technical background to be comfortable using it.  Others have stepped in with
their own apps made with DeOldify, some paid and some free, which is what we want!
We're instead focusing on what we believe we can do best- making better
commercial models that people will pay for.
Does that mean you're not getting the very best for free?  Of course. We simply
don't believe that we're obligated to provide that, nor is it feasible! We
compete on research and sell that.  Not a GUI or web service that wraps said
research- that part isn't something we're going to be great at anyways. We're not
about to shoot ourselves in the foot by giving away our actual competitive
advantage for free, quite frankly.

We're also not willing to go down the rabbit hole of providing endless, open
ended and personalized support on this open source project.  Our position is
this:  If you have the proper background and resources, the project provides
more than enough to get you started. We know this because we've seen plenty of
people using it and making money off of their own projects with it.

Thus, if you have an issue come up and it happens to be an actual bug that
having it be fixed will benefit users generally, then great- that's something
we'll be happy to look into.

In contrast, if you're asking about something that really amounts to asking for
personalized and time consuming support that won't benefit anybody else, we're
not going to help. It's simply not in our interest to do that. We have bills to
pay, after all. And if you're asking for help on something that can already be
derived from the documentation or code?  That's simply annoying, and we're not
going to pretend to be ok with that.
