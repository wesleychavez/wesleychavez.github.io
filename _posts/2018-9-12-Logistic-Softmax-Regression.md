---
layout: post
title: "Logistic / Softmax Regression"
---

Here we’ll explore logistic regression, which is a statistically-based classifier for two classes (dog/no dog), and softmax regression, which is the same classifier but for more than two classes (dog, cat, car, etc.).  One thing to note is that the term “regression” is commonly used to refer to continuous dependent variables (y, the output of our box).  Continuous variables are in opposition to categorical variables that are used in classification.  Continuous variables can take on any number, such as in the task to predict the sale price of the car given its specs.  Here, the dependent variable y can be anywhere from $0-$infinity.  But when we say “logistic regression” or “softmax regression”, we are definitely talking about classification.  Now that’s out of the way,

Logistic regression is an algorithm that, like the perceptron and the SVM, attempts to map an input vector x to an output y, parameterized by weights w.  Instead of the perceptron equation that outputs a binary decision (0 or 1), logistic regression uses the logistic (or sigmoid) function as its activation function:


![Sigmoid](/images/sigmoid.png)

Let’s plot it now.

![Sigmoid Equation](/images/sigmoid_equation.png)

Note that x in the plot corresponds to wx+b in our sigmoid equation.  This equation, like the perceptron, maps the input from 0 to 1, but y in this case is interpreted as a probability that the current example is a positive instance of a class, that is:

![Logistic Probability](/images/logistic_prob.png)

In English, y = “the probability of a dog in the picture, given the current picture and the current weights” (in the dog/no dog task). It makes sense then that you can subtract that probability from 1 and get the probability that there is **no** dog in the picture.

Logistic regression uses the cost function:

![Logistic Cost Function](/images/logistic_cost.png)

where N is the number of training examples and desired is the label/annotation of the current image (0 or 1).  Whoa, what’s going on here?  We know that we want to lower the cost when training the classifier.  Let’s sort this out.  When y = desired = 0 (correct guess) for a given example (n) , our loss is 0 (log of 1 is 0).  Same thing for when y = desired = 1.  Good.  However, if y = 0 and desired = 1, our loss is infinity (okay, technically log(0) is undefined, but as you approach x = 0 from the right (very very small number), -log(x) approaches infinity).  Same for y = 1 and desired = 0.  Of course, y won’t be exactly 0 or 1 in real-world problems.  The basic idea is that our loss is less for values of y that are near desired.  [Taking the derivative](https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression) of the cost function will produce (desired - y)x.  Therefore, we will update every weight as such: 

![Logistic weight update](/images/logistic_update.png)

where ![learning rate](/images/learningrate.png) is the learning rate.  Learning rates are an important machine learning concept, and we will revisit them later.  Basically, when training an algorithm, you don’t want to set the learning rate too low, as it will take forever for the weights to converge to a good solution.  You also don’t want to set it too high, because the weights may overshoot a theoretical optimum if they change too fast.  

Softmax regression is the extension of logistic regression to more than two mutually exclusive classes (dog, cat, car, etc.).  Softmax regression attempts to estimate the probabilities for j classes and replaces the sigmoid function with a softmax function:  

![Softmax function](/images/softmax_function.png)

Notice that after you compute this for all the classes, the sum of all y’s is 1.  This makes sense statistically, because we want to model a probability distribution.  The cost function in this case is 

![Softmax cost](/images/softmax_cost.png)

where desired is again a binary value, 0 for negative classes and 1 for the positive class.  Everything to the right of that summation is also known as “log loss” or “cross-entropy loss”. In order to decrease this function, the update rule is as follows:


![Softmax update](/images/softmax_update.png)

where alpha is the learning rate.

Of course, there is more to training a softmax classifier than what I present here.  Adaptive learning rates exist, which change throughout the training process.  Standardization of your inputs is also important, which I will discuss later.  Also, minibatches (a group of examples) can be used to calculate the weight update equation, saving time, as the equation is not computed for every example.  This post is only to serve as a tool to help grasp the fundamentals of a softmax regression classifier, which is widely used as the last layer in a deep neural network.

I need to tell you about one confusing thing about softmax classifiers: are they linear?  Technically, no, because the softmax function non-linearly transforms the inputs x.  However, let’s say that 

|j     |   wx+b |
|------|--------|
|Dog   |   3    |
|Cat   |   1    |
|Llama |   2.7  |

Computing the softmax function, we get 

|j 	  |   y    | 
|-----|--------|
|Dog  |   0.53 |
|Cat  |   0.07 |
|Llama|   0.39 |

The highest-valued category (dog) before computing the softmax function is *always* going to be the highest-valued category after computation.  This means that after training, we can remove the softmax function to test new examples, and we don’t have to compute the softmax function in order to decide upon the most likely category, we still pick the category with the maximum wx+b.  Wait we only need to compute wx+b? That’s a **linear** equation!!!  So does this mean that only during testing and without the softmax function can the classifier be considered linear?  And is a softmax classifier also drawing linear decision boundaries?  My thoughts are yes to both of these, but since we’re going on a data adventure together, maybe someone can help explain this conundrum.  

Also, please comment if you find errors, or still have gaps in your understanding of logistic/softmax regression.  Next, we’ll win at calculus as we dive into the multi-layer perceptron (MLP), our first non-linear classifier.

