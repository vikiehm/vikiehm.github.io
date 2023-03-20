---
title: "Dominik Muhle"
---

<!-- # Dominik Muhle -->

[LinkedIn](https://de.linkedin.com/in/dominik-muhle-a6b487149)

[GitHub](https://github.com/DominikMuhle)

[Chair Website](https://cvg.cit.tum.de/members/muhled)

## Projects

### Learning Correspondence Uncertainty via Differentiable Nonlinear Least Squares (CVPR 2023)

[Project Page](https://dominikmuhle.github.io/dnls_covs/)

We propose a differentiable nonlinear least squares framework to account for uncertainty in relative pose estimation from feature correspondences. Specifically, we introduce a symmetric version of the probabilistic normal epipolar constraint, and an approach to estimate the covariance of feature positions by differentiating through the camera pose estimation procedure. We evaluate our approach on synthetic, as well as the KITTI and EuRoC real-world datasets. On the synthetic dataset, we confirm that our learned covariances accurately approximate the true noise distribution. In real world experiments, we find that our approach consistently outperforms state-of-the-art non-probabilistic and probabilistic approaches, regardless of the feature extraction algorithm of choice.

### The Probabilistic Normal Epipolar Constraint for Frame-To-Frame Rotation Optimization under Uncertain Feature Positions (CVPR 2022)

[Project Page](https://go.vision.in.tum.de/pnec) | [Paper](https://arxiv.org/abs/2204.02256) | [Code](https://github.com/tum-vision/pnec) | [Video](https://youtu.be/YraCHnR5dmg)

The estimation of the relative pose of two camera views is a fundamental problem in computer vision. Kneip et al. proposed to solve this problem by introducing the normal epipolar constraint (NEC). However, their approach does not take into account uncertainties, so that the accuracy of the estimated relative pose is highly dependent on accurate feature positions in the target frame. In this work, we introduce the probabilistic normal epipolar constraint (PNEC) that overcomes this limitation by accounting for anisotropic and inhomogeneous uncertainties in the feature positions. To this end, we propose a novel objective function, along with an efficient optimization scheme that effectively minimizes our objective while maintaining real-time performance. In experiments on synthetic data, we demonstrate that the novel PNEC yields more accurate rotation estimates than the original NEC and several popular relative rotation estimation algorithms. Furthermore, we integrate the proposed method into a state-of-the-art monocular rotation-only odometry system and achieve consistently improved results for the real-world KITTI dataset.
