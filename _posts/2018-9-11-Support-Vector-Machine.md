---
layout: post
title: "Support Vector Machine"
---

Alright people, I’ll be quick I promise.  Remember the decision boundary of the perceptron?  

![Which one](/images/hyperplane_3.png)

There is a problem with the perceptron.  In this above task, the perceptron may converge upon any one of these solutions, as the perceptron does not update its weights in response to correctly-guessed examples.  Maybe the perceptron could choose a better decision boundary.  Also, in the case of linearly-inseparable data, the perceptron never even converges to a solution, since it is always updating its weights to incorrect examples.  When should the perceptron stop training in this case?

Support Vector Machines (SVMs) solve both of these issues.  The goal of a linear SVM is to maximize the distance between the hyperplane and training instances of both classes.  The support vectors are the input vectors (x) that are closest to the hyperplane. A linear SVM defines hyperplanes such that

![SVM decision boundary](/images/svmdecision.png)

These two equations can be combined into one.

![SVM decision boundary combined](/images/svmdecisioncombined.png)

Alright now for some voodoo.  The distance between a point ![Point](/images/point.png) and a line ![Line](/images/line.png) is ![Distance](/images/distance.png), therefore the distance between a support vector and the hyperplane happens to be ![Distance](/images/dist_hyperplane.png), where ![Euclidean length of w](/images/euclid_w.png) is the Euclidean length of the weights vector.  In order to maximize this distance between support vectors from both classes and the hyperplane, we therefore need to minimize ![Euclidean length of w](/images/euclid_w.png), with the condition that there are no input vectors within the margins. However, when the data is not linearly separable, as in many image classification tasks, we cannot satisfy this condition: 
![Inseparable](/images/svm_inseparable.png)

Thus we introduce the hinge loss function: 

![Hinge loss](/images/hinge.png)

This function is zero if x is on the correct side of the margin, and is proportional to the distance from x to the margin otherwise.  This next equation gets hairy.  SVMs solve the unconstrained optimization problem: 

![SVM optimization](/images/opt.png)

where ![E](/images/e.png) is the hinge loss function I just showed you.  Let’s unpack this.  The SVM minimizes two things: the Euclidean length of the weights vector, and the loss of all N training examples added together, multiplied by C.  The parameter C is a tradeoff parameter that indicates how important it is to classify examples correctly, while sacrificing the ability of the SVM to keep a larger margin between the support vectors and the hyperplane. C can be in the range zero to infinity, and the optimal value for C is different for every problem, so don't forget to try different values. 

But how is an unconstrained optimization problem solved?  The short answer is: quadratic programming.  The long answer requires too much math.  There are many libraries that solve them for you, but my favorite is sklearn.svm.  Also keep in mind that the SVM we are talking about is linear, but we could turn it into a more powerful, non-linear classifier by applying what is known as the “kernel trick”, which we’ll save for another time.
