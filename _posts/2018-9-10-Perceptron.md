The first box I'd like to tell you about is the perceptron.  The perceptron [1], or single-layer perceptron, is one of the most simple forms of artificial neural networks. It is an algorithm for learning a linear, binary classifier to map an input vector x to an output binary value y (in the case of distinguishing between two classes).  For example, say we are trying to differentiate between images of dingos:

![Dingo](/images/dingo.png)

and images of Rhodesian Ridgebacks:

![Rhodesian Ridgeback](/images/rhodesianridgeback.png)

This is a tough problem, as both dogs are usually brown.  One thing to note is that dingos have perkier ears, while Rhodesian Ridgebacks tend to have big ol' floppy ears.  Recall in my last post that we need to shove data into the box iteratively (one-by-one) so that the box can learn a little bit from each one.  For the purposes of this example, instead of shoving in each picture as an input (as one would normally do when classifying images), we're going to use the "presence" of a smaller picture as an input feature.  To clarify, say we are using these four smaller pictures as features: 

![Features](/images/features.png)

Note that the two on the left are taken directly from the picture of the dingo.  We want our perceptron to learn a mapping of the "presence" of these features to whether or not we are looking at a picture of a dingo.  Alright, let's introduce some math.  The perceptron "box" equation is: 

![Perceptron Equation](/images/perceptron.png)

where w is the weight vector, and b is the bias. w and b are the knobs to tweak, x is the input vector (presence of the smaller pictures), and y is the output binary value.  If y is 1, the current picture is a dingo.  If y is 0, the perceptron is confident that the current picture is a Rhodesian Ridgeback.  

Let's pretend that we've already trained the perceptron (we finished tweaking the knobs after showing it thousands of examples of dingos and Ridgebacks) and it's super accurate at guessing now.  Let's try to classify the first dingo picture:

![Perceptron Computation](/images/computation.png)

Since the first two features are present in the picture, the input for those features are one.  (In a real world scenario, we will have pictures that don't have the same exact pointy ears as the first feature.  Maybe we will find a similar pointy ear in a picture, and x will be 0.9.  It all depends on the feature extractor you use.)  Solving the perceptron equation, wx+b = 1(0.8) + 1(0.7) + 0(-0.4) + 0(-0.4) + 0(0) = 1.5, which is greater than zero, so y = 1, and this picture is correctly classified as a dingo.

But how did we tweak the knobs correctly to decide these final weights?  Great question!  All machine learning models have a cost function, sometimes referred to as a loss, energy, or optimization function.  There are little differences in these terms, but for now I'll lump them together.  The cost function is a metric for how well your model classifies.  The perceptron cost function is:

![Perceptron Cost](/images/perceptroncost.png)

"desired" is the ground truth annotation (remember the sticky notes) for each labeled picture.  If "desired" is the same as y (the perceptron's guess for a particular picture), this is good, and the loss for that particular training example is 0.  If the perceptron guesses wrong, the loss will be one.  All the losses over the whole dataset added together is the perceptron's cost, which we want to minimize.

So how do we minimize this cost?  By tweaking the knobs (w and b vectors).  After each training vector (x) is presented, the perceptron makes a guess (y), and this is compared to the ground truth annotation (desired).  The weights (w) are then updated:

![Perceptron weight update rule](/images/perceptronweightupdate.png)

If the perceptron misclassifies an example, this update moves the weights in the right direction (the direction in which the classifier is more likely to classify this example correctly the next time it is input to the perceptron). If the perceptron makes a correct prediction for an example, the network weights do not change. This is important, as the perceptron only optimizes the network to correctly classify as many training examples as possible.  In order to extend the perceptron to more than two classes (dingo vs. Rhodesian Ridgeback vs. poodle), the perceptron uses j weight vectors, where j is the number of classes (3), and chooses a class according to:

![Perceptron equation, multiclass](/images/perceptronmulticlass.png)

This picks the class (j) that has the highest value of wx+b.  Note that y is no longer binary. If desired = y, all weights are unchanged. Otherwise, the update rule becomes:

![Perceptron multiclass weight update](/images/perceptronmulticlassweightupdate.png)

This form of multiclass classification is known as “one vs. all”, where there exists one weight vector for every class, and each perceptron attempts to separate data from the corresponding class and data from all other classes.  In the dingo vs. Rhodesian Ridgeback vs. poodle example, there will essentially be three perceptrons, each differentiating between: dingo/not dingo, Ridgeback/not Ridgeback, and poodle/not poodle.  This is in opposition to a strategy called "one vs. one", where, in our example, there would be still be 3 perceptrons, but they differentiate between: dingo vs. Ridgeback, Ridgeback vs. poodle, and poodle vs. dingo. However, as the number of classes (N) increases, the number of "one vs. one" perceptrons increases, as there are N(N-1)/2 classifiers.

[1] F. Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6):386, 1958.
